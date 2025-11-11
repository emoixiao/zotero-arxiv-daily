import arxiv
import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pyzotero import zotero
from recommender import rerank_paper
from construct_email import render_email, send_email
from tqdm import trange,tqdm
from loguru import logger
from gitignore_parser import parse_gitignore
from tempfile import mkstemp
from paper import ArxivPaper
from llm import set_global_llm
import feedparser

def get_zotero_corpus(id:str,key:str) -> list[dict]:
    zot = zotero.Zotero(id, 'user', key)
    collections = zot.everything(zot.collections())
    collections = {c['key']:c for c in collections}
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    def get_collection_path(col_key:str) -> str:
        if p := collections[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections[col_key]['data']['name']
        else:
            return collections[col_key]['data']['name']
    for c in corpus:
        paths = [get_collection_path(col) for col in c['data']['collections']]
        c['paths'] = paths
    return corpus

def filter_corpus(corpus:list[dict], pattern:str) -> list[dict]:
    _,filename = mkstemp()
    with open(filename,'w') as file:
        file.write(pattern)
    matcher = parse_gitignore(filename,base_dir='./')
    new_corpus = []
    for c in corpus:
        match_results = [matcher(p) for p in c['paths']]
        if not any(match_results):
            new_corpus.append(c)
    os.remove(filename)
    return new_corpus


def get_arxiv_paper_from_rss(query: str, debug: bool = False) -> List[ArxivPaper]:
    """直接从 RSS 源获取论文信息，避免调用 arXiv API"""
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    
    if 'Feed error for query' in feed.feed.title:
        raise Exception(f"Invalid ARXIV_QUERY: {query}.")
    
    papers = []
    entries = [i for i in feed.entries if i.arxiv_announce_type == 'new']
    
    if debug:
        entries = entries[:5]  # 调试模式下只取5篇
    
    logger.info(f"Processing {len(entries)} papers from RSS feed")
    
    for entry in tqdm(entries, desc="Processing RSS entries"):
        try:
            # 从 RSS 条目创建论文对象
            paper_data = {
                'entry_id': entry.id,
                'title': entry.title,
                'summary': entry.summary,
                'published': entry.published,
                'authors': [author.name for author in entry.authors],
                'arxiv_primary_category': getattr(entry, 'arxiv_primary_category', {}).get('term', ''),
                'arxiv_comment': getattr(entry, 'arxiv_comment', ''),
                'links': entry.links
            }
            
            # 这里需要修改 ArxivPaper 类来支持从字典创建
            # 或者创建一个临时解决方案
            paper = create_paper_from_rss_data(paper_data)
            papers.append(paper)
            
            # 处理每个条目后的小延迟
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing entry {entry.id}: {e}")
            continue
    
    return papers

def create_paper_from_rss_data(paper_data: dict) -> ArxivPaper:
    """从 RSS 数据创建 ArxivPaper 对象"""
    # 这是一个临时解决方案，你需要根据你的 ArxivPaper 类来调整
    # 如果 ArxivPaper 需要 arxiv.Result 对象，这可能比较困难
    # 但至少可以获取基本信息用于推荐
    
    # 临时方案：创建一个简单的对象
    class SimplePaper:
        def __init__(self, data):
            self.entry_id = data['entry_id']
            self.title = data['title']
            self.summary = data['summary']
            self.published = data['published']
            self.authors = data['authors']
            self.primary_category = data['arxiv_primary_category']
            self.comment = data['arxiv_comment']
            
    return ArxivPaper(SimplePaper(paper_data))  # 可能需要调整 ArxivPaper 的构造函数



parser = argparse.ArgumentParser(description='Recommender system for academic papers')

def add_argument(*args, **kwargs):
    def get_env(key:str,default=None):
        # handle environment variables generated at Workflow runtime
        # Unset environment variables are passed as '', we should treat them as None
        v = os.environ.get(key)
        if v == '' or v is None:
            return default
        return v
    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get('dest',args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        #convert env_value to the specified type
        if kwargs.get('type') == bool:
            env_value = env_value.lower() in ['true','1']
        else:
            env_value = kwargs.get('type')(env_value)
        parser.set_defaults(**{arg_full_name:env_value})


if __name__ == '__main__':
    
    add_argument('--zotero_id', type=str, help='Zotero user ID')
    add_argument('--zotero_key', type=str, help='Zotero API key')
    add_argument('--zotero_ignore',type=str,help='Zotero collection to ignore, using gitignore-style pattern.')
    add_argument('--send_empty', type=bool, help='If get no arxiv paper, send empty email',default=False)
    add_argument('--max_paper_num', type=int, help='Maximum number of papers to recommend',default=100)
    add_argument('--arxiv_query', type=str, help='Arxiv search query')
    add_argument('--smtp_server', type=str, help='SMTP server')
    add_argument('--smtp_port', type=int, help='SMTP port')
    add_argument('--sender', type=str, help='Sender email address')
    add_argument('--receiver', type=str, help='Receiver email address')
    add_argument('--sender_password', type=str, help='Sender email password')
    add_argument(
        "--use_llm_api",
        type=bool,
        help="Use OpenAI API to generate TLDR",
        default=False,
    )
    add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        default=None,
    )
    add_argument(
        "--openai_api_base",
        type=str,
        help="OpenAI API base URL",
        default="https://api.openai.com/v1",
    )
    add_argument(
        "--model_name",
        type=str,
        help="LLM Model Name",
        default="gpt-4o",
    )
    add_argument(
        "--language",
        type=str,
        help="Language of TLDR",
        default="English",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    assert (
        not args.use_llm_api or args.openai_api_key is not None
    )  # If use_llm_api is True, openai_api_key must be provided
    if args.debug:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
        logger.debug("Debug mode is on.")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.info("Retrieving Zotero corpus...")
    corpus = get_zotero_corpus(args.zotero_id, args.zotero_key)
    logger.info(f"Retrieved {len(corpus)} papers from Zotero.")
    if args.zotero_ignore:
        logger.info(f"Ignoring papers in:\n {args.zotero_ignore}...")
        corpus = filter_corpus(corpus, args.zotero_ignore)
        logger.info(f"Remaining {len(corpus)} papers after filtering.")
    logger.info("Retrieving Arxiv papers...")
    papers = get_arxiv_paper(args.arxiv_query, args.debug)
    if len(papers) == 0:
        logger.info("No new papers found. Yesterday maybe a holiday and no one submit their work :). If this is not the case, please check the ARXIV_QUERY.")
        if not args.send_empty:
          exit(0)
    else:
        logger.info("Reranking papers...")
        papers = rerank_paper(papers, corpus)
        if args.max_paper_num != -1:
            papers = papers[:args.max_paper_num]
        if args.use_llm_api:
            logger.info("Using OpenAI API as global LLM.")
            set_global_llm(api_key=args.openai_api_key, base_url=args.openai_api_base, model=args.model_name, lang=args.language)
        else:
            logger.info("Using Local LLM as global LLM.")
            set_global_llm(lang=args.language)

    html = render_email(papers)
    logger.info("Sending email...")
    send_email(args.sender, args.receiver, args.sender_password, args.smtp_server, args.smtp_port, html)
    logger.success("Email sent successfully! If you don't receive the email, please check the configuration and the junk box.")


import arxiv
import time
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any
from typing import List, Dict, Optional
from functools import wraps
from config import setting

logger = logging.getLogger(__name__)

def limit(func):
    """arxiv query 的速率限制"""
    last_request_time = {"time": 0}

    @wraps(func)
    def wrapper(*args, **kwargs):
        elapsed = time.time() - last_request_time["time"]
        if elapsed < setting.rate_limit:
            sleep_time = setting.rate_limit - elapsed
            logger.debug(f"[速率限制] 等待 {sleep_time:.2f}秒")
            time.sleep(sleep_time)

        last_request_time["time"] = time.time()
        return func(*args, **kwargs)

    return wrapper

class ArxivQuery:
    """Arxiv 论文查询工具"""
    def __init__(self, rate_limit: Optional[float] = None, retry_times = None):
        """
        初始化 Arxiv 查询工具
        """
        self.rate_limit = rate_limit or setting.rate_limit
        self.retry_times = retry_times or setting.max_retries

        logger.info(f"正在初始化 Arxiv 工具.")


    @limit
    def search_papers(self,
        query: Optional[str] = None,
        *,
        title: Optional[str] = None,
        author: Optional[str] = None,
        abstract: Optional[str] = None,
        category: Optional[str] = None,
        ids: Optional[List[str] | tuple[str, ...] | str] = None,
        start: int = 0,
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ) -> List[Dict]:
        """检索 arXiv 论文，并返回结构化论文元数据列表。

        当用户想查找论文、相关工作、某个作者的论文、某个方向的最新论文、
        某个 arXiv 分类下的论文，或根据 arXiv id 获取论文信息时，使用这个工具。
        最终回答必须基于本工具返回的论文结果，不要编造标题、作者、链接、时间或摘要。

        查询方式：
        - 普通关键词检索：使用 ``query``，例如 ``"gaussian splatting"``。
        - 原生 arXiv 查询语法：也使用 ``query``，例如
          ``"all:electron AND cat:cs.CV"``、``"ti:diffusion AND au:Ho"``。
        - 精确字段检索：优先使用 ``title``、``author``、``abstract``、``category``。
          同时提供多个字段时，会用 AND 组合条件。
        - 按 id 查询：如果用户提供明确的 arXiv id，优先使用 ``ids``。
          ``ids`` 可以是单个 id、逗号分隔字符串或 id 列表。

        Args:
            query: 普通关键词或原生 arXiv query API 语法。普通关键词会自动转为
                ``all:<query>`` 在所有字段中检索。
            title: 只在论文标题中搜索的关键词或短语。
            author: 按作者姓名搜索，例如 ``"Yann LeCun"``。
            abstract: 只在论文摘要中搜索的关键词或短语。
            category: arXiv 分类，例如 ``"cs.CV"``、``"cs.LG"``、``"stat.ML"``。
            ids: 单个 arXiv id，逗号分隔的多个 id，或 id 列表。
            start: 从 0 开始的结果偏移量，用于分页。
            max_results: 最多返回的论文数量。
            sort_by: 排序字段。常用值包括 ``arxiv.SortCriterion.Relevance``、
                ``arxiv.SortCriterion.SubmittedDate``、``arxiv.SortCriterion.LastUpdatedDate``。
            sort_order: 排序方向。常用值包括 ``arxiv.SortOrder.Descending`` 和
                ``arxiv.SortOrder.Ascending``。查最新论文时通常使用 Descending。

        Returns:
            论文元数据列表。每篇论文通常包含：
            ``arxiv_id``、``entry_id``、``abs_url``、``title``、``authors``、
            ``abstract``、``summary``、``published``、``updated``、``categories``、
            ``primary_category``、``pdf_url``、``doi``、``journal_ref``、``comment``、
            ``links``。
        """
        logger.info(f"正在调用 arxiv 搜索工具链，")

        try:
            if start < 0:
                raise ValueError("start 必须 >= 0")
            if max_results < 1:
                raise ValueError("max_results 必须 >= 1")

            built_query = self._build_search_query(
                search_query=query,
                title=title,
                author=author,
                abstract=abstract,
                category=category,
            )
            id_list = self._format_id_list(ids)

            if not built_query and not id_list:
                raise ValueError("请至少提供一个 search condition 或 ids")

            max_allowed = getattr(setting, "arxiv_max_results", max_results)
            result_limit = min(max_results, max_allowed)
            fetch_limit = start + result_limit

            search = arxiv.Search(
                query=built_query,
                id_list=id_list,
                max_results=fetch_limit,
                sort_by=sort_by,
                sort_order=sort_order,
            )

            results = self._get_results(search, page_size=min(fetch_limit, 100))

            papers = [self._paper_to_metadata(paper) for paper in list(results)[start:]]

            logger.info(f"成功获取 {len(papers)} 篇论文")
            return papers

        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise


    @limit
    def download_pdf(self,
        arxiv_id: Optional[str] = None,
        save_dir: Optional[str] = None,
    ):
        """根据 arXiv id 下载论文 PDF，并返回本地 PDF 文件路径。

        当用户明确要求下载某篇 arXiv 论文的 PDF、保存论文全文、获取本地 PDF 文件，
        或后续需要对论文全文做解析和阅读时，使用这个工具。
        调用前应尽量确认用户提供的是明确的 arXiv id；如果用户只给了论文标题、
        作者或研究方向，应先调用 ``search_papers`` 检索论文，再选择对应结果的
        ``arxiv_id`` 调用本工具。

        Args:
            arxiv_id: 要下载的 arXiv id，例如 ``"2308.04079"``、
                ``"2308.04079v1"`` 或旧格式 id。不要传论文标题、URL、作者名或关键词。
            save_dir: PDF 保存目录。为空时使用配置中的默认目录
                ``setting.pdf_save_dir``。

        Returns:
            下载成功时返回本地 PDF 文件路径字符串，例如
            ``"./database/Papers/2308.04079.pdf"``。
            如果论文不存在、下载失败或重试后仍失败，返回 ``None``。

        Notes:
            本工具只负责下载 PDF，不负责总结、解析或问答。下载完成后如需阅读论文内容，
            应把返回的本地路径交给后续 PDF 解析或 RAG 工具。
        """
        pdf_save_dir = Path(save_dir or setting.pdf_save_dir)
        pdf_save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"正在下载PDF{arxiv_id}")

        for attempt in range(self.retry_times):
            try:
                search = arxiv.Search(id_list=[arxiv_id])
                result = next(self._get_results(search, page_size=1))

                pdf_path = pdf_save_dir / f"{arxiv_id}.pdf"
                urllib.request.urlretrieve(result.pdf_url, pdf_path)
                logger.info(f"PDF下载成功: {pdf_path}")
                return str(pdf_path)
            
            except StopIteration:
                logger.error(f"[❌错误] 下载失败，论文不存在: {arxiv_id}")
                return None

            except Exception as e:
                logger.warning(f"[⚠️警告] 下载尝试 {attempt + 1} 失败: {str(e)}")
                if attempt < self.retry_times - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    logger.error(f"[❌错误] 下载失败，已重试 {self.retry_times} 次")
                    return None


    @limit
    def get_paper_metadata(self, arxiv_id: Optional[str] = None) -> Optional[Dict]:
        """根据 arXiv id 获取单篇论文的结构化元数据。

        当用户已经提供明确的 arXiv id，并希望查看论文标题、作者、摘要、发布时间、
        PDF 链接、分类、DOI、期刊引用或其他基本信息时，使用这个工具。
        如果用户只提供论文标题、作者、关键词或研究方向，应先调用 ``search_papers``
        检索候选论文，再选择对应结果的 ``arxiv_id`` 调用本工具。
        最终回答必须基于本工具返回的元数据，不要编造论文信息。

        Args:
            arxiv_id: 要查询的 arXiv id，例如 ``"2308.04079"``、
                ``"2308.04079v1"`` 或旧格式 id。不要传论文标题、URL、作者名或关键词。

        Returns:
            找到论文时返回元数据字典，字段与 ``search_papers`` 的单篇论文结果一致，
            通常包含 ``arxiv_id``、``entry_id``、``abs_url``、``title``、``authors``、
            ``abstract``、``summary``、``published``、``updated``、``categories``、
            ``primary_category``、``pdf_url``、``doi``、``journal_ref``、``comment``、
            ``links``。如果 ``arxiv_id`` 为空、论文不存在或查询失败，返回 ``None``。
        """
        if not arxiv_id or not arxiv_id.strip():
            logger.error("获取元数据失败: arxiv_id 不能为空")
            return None

        arxiv_id = arxiv_id.strip()
        logger.info(f"获取论文元数据: {arxiv_id}")

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = self._get_results(search, page_size=1)

            paper = next(results)
            metadata = self._paper_to_metadata(paper)

            logger.info(f"成功获取元数据: {metadata['title']}")
            return metadata

        except StopIteration:
            logger.error(f"[❌错误] 论文不存在: {arxiv_id}")
            return None
        except Exception as e:
            logger.error(f"[❌错误] 获取元数据失败: {str(e)}")
            return None


    def _paper_to_metadata(self, paper) -> Dict:
        return {
            "arxiv_id": paper.get_short_id(),
            "entry_id": paper.entry_id,
            "abs_url": paper.entry_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "summary": paper.summary,
            "published": paper.published.isoformat(),
            "updated": paper.updated.isoformat(),
            "categories": paper.categories,
            "primary_category": paper.primary_category,
            "pdf_url": paper.pdf_url,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comment": paper.comment,
            "links": [
                {"href": paper.entry_id, "rel": "alternate", "type": "text/html"},
                {"href": paper.pdf_url, "rel": "related", "type": "application/pdf", "title": "pdf"},
            ],
        }


    def _get_results(self, search, page_size: int = 100):
        client_cls = getattr(arxiv, "Client", None)
        if client_cls is not None:
            client = client_cls(
                page_size=page_size,
                delay_seconds=self.rate_limit,
                num_retries=self.retry_times,
            )
            return client.results(search)
        return search.results()


    def _build_search_query(self,
        *,
        search_query: Optional[str],
        title: Optional[str],
        author: Optional[str],
        abstract: Optional[str],
        category: Optional[str],
    ) -> str:
        parts: List[str] = []

        if search_query:
            search_query = search_query.strip()
            parts.append(search_query if self._looks_like_arxiv_query(search_query) else self._term("all", search_query))
        if title:
            parts.append(self._term("ti", title))
        if author:
            parts.append(self._term("au", author))
        if abstract:
            parts.append(self._term("abs", abstract))
        if category:
            parts.append(self._term("cat", category, quote=False))

        return " AND ".join(parts)


    def _looks_like_arxiv_query(self, value: str) -> bool:
        return bool(re.search(r"\b(?:all|ti|au|abs|co|jr|cat|rn|id):", value))


    def _term(self, prefix: str, value: str, *, quote: bool = True) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{prefix} query 不能为空")
        if quote and (" " in value or ":" in value):
            value = value.replace('"', r"\"")
            return f'{prefix}:"{value}"'
        return f"{prefix}:{value}"


    def _format_id_list(self, ids: Optional[List[str] | tuple[str, ...] | str]) -> List[str]:
        if ids is None:
            return []
        if isinstance(ids, str):
            return [item.strip() for item in ids.split(",") if item.strip()]
        return [item.strip() for item in ids if item.strip()]

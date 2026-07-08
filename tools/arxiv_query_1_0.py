from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
import urllib.error
import urllib.parse
import urllib.request as libreq
import time
from time import sleep

ARXIV_API_URL = "http://export.arxiv.org/api/query"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"

def query(
    search_query: str | None = None,
    *,
    title: str | None = None,
    author: str | None = None,
    abstract: str | None = None,
    category: str | None = None,
    ids: list[str] | tuple[str, ...] | str | None = None,
    start: int = 0,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    timeout: int = 60,
    retry_times: int = 3
) -> dict[str, Any]:
    """按关键词、标题、作者、摘要、分类或 arXiv id 查询 arXiv papers。

    当用户想查找 arXiv 论文时使用这个工具。
    简单关键词或短语检索使用 ``search_query``。
    明确限定字段时，优先使用 ``title``、``author``、``abstract`` 或 ``category``。
    同时传入多个搜索字段时，这些条件会用 AND 组合。
    如果用户提供了明确的 arXiv id，优先使用 ``ids``。

    Args:
        search_query: 原始 arXiv query API，例如 ``all:electron AND cat:cs.CV``；
            也可以是普通关键词或短语，此时会在所有字段中搜索。
        title: 只在论文标题中搜索。
        author: 按作者姓名搜索。
        abstract: 只在论文摘要中搜索。
        category: arXiv 分类，例如 ``cs.CV`` 或 ``stat.ML``。
        ids: 单个 arXiv id，或多个 arXiv id 组成的列表。
        start: 从 0 开始的结果偏移量。
        max_results: 最多返回的论文数量。
        sort_by: 排序字段，只能是 ``relevance``、``lastUpdatedDate`` 或 ``submittedDate``。
        sort_order: 排序方向，只能是 ``ascending`` 或 ``descending``。
        timeout: 请求超时时间，单位为秒。
        retry_times： 请求超时后重试次数，单位为次。

    Returns:
        包含 feed metadata 和 ``papers`` 列表的 dict。
        ``papers`` 列表中的每篇论文包含 ``id``、``title``、``summary``、
        ``authors``、``published``、``updated``、``categories``、
        ``primary_category``、``comment``、``journal_ref``、``doi``、
        ``abs_url``、``pdf_url`` 和 ``links``。
    """
    print("正在调用arxiv工具链...")
    if start < 0:
        raise ValueError("start 必须 >= 0")
    if max_results < 1:
        raise ValueError("max_results 必须 >= 1")

    sort_by = _validate_choice(
        sort_by,
        {"relevance", "lastUpdatedDate", "submittedDate"},
        "sort_by",
    )
    sort_order = _validate_choice(
        sort_order,
        {"ascending", "descending"},
        "sort_order",
    )

    built_query = _build_search_query(
        search_query=search_query,
        title=title,
        author=author,
        abstract=abstract,
        category=category,
    )
    
    id_list = _format_id_list(ids)

    if not built_query and not id_list:
        raise ValueError("请至少提供一个 search condition 或 ids")

    params: dict[str, Any] = {
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    if built_query:
        params["search_query"] = built_query
    if id_list:
        params["id_list"] = id_list

    if retry_times < 1:
        raise ValueError("retry_times 必须 >= 1")

    request_url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    for i in range(retry_times):
        try:
            with libreq.urlopen(request_url, timeout=timeout) as url:
                response_text = url.read().decode("utf-8")
            break
        except TimeoutError:
            if i == retry_times - 1:
                raise
            time.sleep(2 ** i)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) and i < retry_times - 1:
                time.sleep(2 ** i)
                continue
            raise
    sleep(3) # 请求 “one request every three seconds”，防止 “Rate Exceed”
    return _parse_response(response_text)

LegacyArxivQueryTools = [query]


def _build_search_query(
    *,
    search_query: str | None,
    title: str | None,
    author: str | None,
    abstract: str | None,
    category: str | None,
) -> str:
    parts: list[str] = []

    if search_query:
        search_query = search_query.strip()
        parts.append(search_query if _looks_like_arxiv_query(search_query) else _term("all", search_query))
    if title:
        parts.append(_term("ti", title))
    if author:
        parts.append(_term("au", author))
    if abstract:
        parts.append(_term("abs", abstract))
    if category:
        parts.append(_term("cat", category, quote=False))

    return " AND ".join(parts)

def _looks_like_arxiv_query(value: str) -> bool:
    return bool(re.search(r"\b(?:all|ti|au|abs|co|jr|cat|rn|id):", value))

def _term(prefix: str, value: str, *, quote: bool = True) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{prefix} query 不能为空")
    if quote and (" " in value or ":" in value):
        value = value.replace('"', r"\"")
        return f'{prefix}:"{value}"'
    return f"{prefix}:{value}"

def _format_id_list(ids: list[str] | tuple[str, ...] | str | None) -> str:
    if ids is None:
        return ""
    if isinstance(ids, str):
        return ids.strip()
    return ",".join(item.strip() for item in ids if item.strip())

def _parse_response(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)

    return {
        "title": _text(root, f"{ATOM_NS}title"),
        "updated": _text(root, f"{ATOM_NS}updated"),
        "total_results": _int_text(root, f"{OPENSEARCH_NS}totalResults"),
        "start_index": _int_text(root, f"{OPENSEARCH_NS}startIndex"),
        "items_per_page": _int_text(root, f"{OPENSEARCH_NS}itemsPerPage"),
        "papers": [_parse_entry(entry) for entry in root.findall(f"{ATOM_NS}entry")],
    }

def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    links = [_parse_link(link) for link in entry.findall(f"{ATOM_NS}link")]
    categories = [
        category.attrib["term"]
        for category in entry.findall(f"{ATOM_NS}category")
        if category.attrib.get("term")
    ]
    primary_category = entry.find(f"{ARXIV_NS}primary_category")

    return {
        "id": _arxiv_id(_text(entry, f"{ATOM_NS}id")),
        "title": _clean_text(_text(entry, f"{ATOM_NS}title")),
        "summary": _clean_text(_text(entry, f"{ATOM_NS}summary")),
        "authors": [
            _text(author, f"{ATOM_NS}name")
            for author in entry.findall(f"{ATOM_NS}author")
        ],
        "published": _text(entry, f"{ATOM_NS}published"),
        "updated": _text(entry, f"{ATOM_NS}updated"),
        "categories": categories,
        "primary_category": primary_category.attrib.get("term") if primary_category is not None else None,
        "comment": _clean_text(_text(entry, f"{ARXIV_NS}comment")),
        "journal_ref": _clean_text(_text(entry, f"{ARXIV_NS}journal_ref")),
        "doi": _text(entry, f"{ARXIV_NS}doi"),
        "abs_url": _first_link(links, rel="alternate"),
        "pdf_url": _first_link(links, title="pdf"),
        "links": links,
    }

def _parse_link(link: ET.Element) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "href": link.attrib.get("href"),
            "rel": link.attrib.get("rel"),
            "type": link.attrib.get("type"),
            "title": link.attrib.get("title"),
        }.items()
        if value is not None
    }

def _first_link(
    links: list[dict[str, str]],
    *,
    rel: str | None = None,
    title: str | None = None,
) -> str | None:
    for link in links:
        if rel is not None and link.get("rel") != rel:
            continue
        if title is not None and link.get("title") != title:
            continue
        return link.get("href")
    return None

def _text(element: ET.Element, path: str) -> str:
    found = element.find(path)
    return found.text.strip() if found is not None and found.text else ""

def _int_text(element: ET.Element, path: str) -> int:
    value = _text(element, path)
    return int(value) if value else 0

def _clean_text(value: str) -> str:
    return " ".join(value.split())

def _arxiv_id(value: str) -> str:
    return value.rsplit("/abs/", 1)[-1] if value else ""

def _validate_choice(value: str, choices: set[str], name: str) -> str:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} 必须是以下值之一: {allowed}")
    return value

if __name__ == "__main__":
    result = query(search_query="DeferredGS", author="Lin Gao")
    print(result)

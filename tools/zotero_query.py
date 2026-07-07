from __future__ import annotations

import hashlib
import logging
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from pyzotero import Zotero
from webdav3.client import Client

from config import setting

from .lazy import LazyToolList

logger = logging.getLogger(__name__)

class ZoteroClient:
    def __init__(self,
        library_type: str = "user",
        trust_env: bool = True,
    ):
        """创建 Zotero client"""
        logger.info("正在初始化 Zotero client，library_type=%s", library_type)
        if not setting.ZoteroID or not setting.ZoteroKeys:
            logger.error("Zotero client 初始化失败: ZoteroID 或 ZoteroKeys 缺失")
            raise ValueError("ZoteroID and ZoteroKeys 缺一不可")

        self.zoteroclient = self._CreateZoteroClient(library_type)
        self.webdavclient = self._CreateWebDavClient(trust_env)
        logger.info("Zotero client 初始化完成")

    def _CreateWebDavClient(self,
        trust_env: bool
    ) -> Client:
        logger.info("正在创建 WebDAV client，trust_env=%s", trust_env)
        WebDav_Option = {
            "webdav_hostname": setting.WEBDAV_HOSTNAME,
            "webdav_login": setting.WEBDAV_LOGIN,
            "webdav_password": setting.WEBDAV_PASSWD,
            }
        client = Client(WebDav_Option)
        client.session.trust_env = trust_env
        logger.info("WebDAV client 创建完成")
        return client

    def _CreateZoteroClient(self, library_type: str = "user") -> Zotero:
        logger.info("正在创建 Zotero API client，library_type=%s", library_type)
        return Zotero(
            library_id=setting.ZoteroID,
            api_key=setting.ZoteroKeys,
            library_type=library_type,
        )

    def SearchOnParentTitle(self,
        title: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """按标题、作者或年份搜索 Zotero 文献库中的顶级条目。

        当用户想查找已经保存到 Zotero 文献库中的论文、书籍或参考文献时，
        使用这个工具。用户可能会提供完整标题、部分标题、作者名、年份，
        或论文题目中的关键词。本工具只搜索 Zotero 顶级父条目，不返回
        子附件信息。

        Args:
            title: 用于 Zotero title/creator/year 查询模式的搜索文本。
                可以是完整论文标题、部分标题、作者名、年份或参考文献中的短语。
            limit: 最多返回的匹配父条目数量。

        Returns:
            父条目元数据列表。每个结果可能包含 ``key``、``title``、
            ``itemType``、``authors``、``abstractNote``、``archiveID``、
            ``extra``、``DOI``、``url`` 和 ``tags``。如需继续查询该条目
            下的附件，应将返回结果中的 ``key`` 作为 ``ParentKey`` 传给
            ``GetParentsChildrenMetadatas``。
        """
        if not title.strip():
            logger.error("Zotero 父条目搜索失败: title 为空")
            raise ValueError("标题不为空！！")
        search_title = title.strip()
        logger.info("正在搜索 Zotero 父条目，query=%s，limit=%s", search_title, limit)
        results = self.zoteroclient.top(q=search_title, qmode="titleCreatorYear", limit=limit)
        Metadatas = [self.ProcessParentMetadata(result) for result in results]
        logger.info("Zotero 父条目搜索完成，返回 %s 条结果", len(Metadatas))
        return Metadatas
    
    def GetParentMetadata(self,
        ParentKey: str | int,
    ) -> dict[str, Any]:
        """通过 Key 搜索 zotero 数据库中顶级条目"""
        logger.info("正在获取 Zotero 父条目元数据，ParentKey=%s", ParentKey)
        ParentResults = self.zoteroclient.item(ParentKey)
        Metadata = self.ProcessParentMetadata(ParentResults)
        logger.info("Zotero 父条目元数据获取完成，ParentKey=%s", ParentKey)
        return Metadata
    
    def ProcessParentMetadata(self,
        ParentItem: dict[str, Any]
    ) -> dict[str, Any]:
        ParentData = ParentItem.get("data", {})

        Metadata = {
            'key': ParentData.get('key'),
            'title': ParentData.get('title'),
            'itemType': ParentData.get('itemType'),
            'authors': ParentData.get('creators', []),
            'abstractNote': ParentData.get('abstractNote'),
            'archiveID': ParentData.get('archiveID'),
            'extra': ParentData.get('extra'),
            'DOI': ParentData.get('DOI'),
            'url': ParentData.get('url'),
            'tags': ParentData.get('tags'),
        }
        return Metadata

    def GetParentsChildrenMetadatas(self,
        ParentKey: str | int
    ) -> list[dict[str, Any]]:
        """获取某个 Zotero 父条目下的子条目元数据。

        当已经通过 ``SearchOnParentTitle`` 找到 Zotero 父条目，并且用户
        需要查看、定位或下载该条目下的附件文件时，使用这个工具。Zotero
        中论文 PDF 通常作为父条目的子附件条目保存。

        Args:
            ParentKey: Zotero 顶级父条目的 key。这个值应来自
                ``SearchOnParentTitle`` 返回结果中的 ``key``，或其他父条目
                元数据查询结果；不要传入子附件条目的 key。

        Returns:
            子条目元数据列表。每个子条目可能包含 ``key``、``parentItem``、
            ``itemType``、``contentType``、``filename``、``md5`` 和
            ``tags``。如果子条目是 PDF 附件，应将该子条目的 ``key`` 作为
            ``ChildrenKey`` 传给 ``GetChildrenAttachment`` 下载附件。
        """
        logger.info("正在获取 Zotero 子条目元数据，ParentKey=%s", ParentKey)
        ChildrenResults = self.zoteroclient.children(ParentKey)
        ChildrenMetadatas = [self.ProcessChildrenMetadata(ChildrenResult) for ChildrenResult in ChildrenResults]
        logger.info(
            "Zotero 子条目元数据获取完成，ParentKey=%s，返回 %s 条结果",
            ParentKey,
            len(ChildrenMetadatas),
        )
        return ChildrenMetadatas

    def GetChildrenMetadata(self,
        ChildrenKey: str | int,
    ) -> dict[str, Any]:
        logger.info("正在获取 Zotero 子条目元数据，ChildrenKey=%s", ChildrenKey)
        ChildrenResults = self.zoteroclient.item(ChildrenKey)
        Metadatas = self.ProcessChildrenMetadata(ChildrenResults)
        logger.info("Zotero 子条目元数据获取完成，ChildrenKey=%s", ChildrenKey)
        return Metadatas
    
    def ProcessChildrenMetadata(self,
        ChildrenItem: dict[str, Any]
    ) -> dict[str, Any]:
        ChildrenData = ChildrenItem.get("data", {})

        Metadata = {
            'key': ChildrenData.get('key'),
            'parentItem': ChildrenData.get('parentItem'),
            'itemType': ChildrenData.get('itemType'),
            'contentType': ChildrenData.get('contentType'),
            'filename': ChildrenData.get('filename'),
            'md5': ChildrenData.get('md5'),
            'tags': ChildrenData.get('tags'),
        }
        return Metadata

    def _DownloadWebDavFile(
        self,
        remote_path: str,
        local_path: Path,
    ) -> Path:
        if local_path.is_file():
            logger.info("WebDAV 文件已存在，跳过下载: %s", local_path)
            return local_path

        logger.info("正在下载 WebDAV 文件，remote_path=%s，local_path=%s", remote_path, local_path)
        self.webdavclient.download_sync(
            remote_path=remote_path,
            local_path=str(local_path),
        )
        logger.info("WebDAV 文件下载完成: %s", local_path)
        return local_path

    def _CalculateZipPdfMD5(self, zip_path: Path) -> str:
        """计算 Zotero WebDAV zip 包内 PDF 文件的 MD5。"""
        logger.info("正在计算 zip 内 PDF 的 MD5: %s", zip_path)
        file_hash = hashlib.md5()

        try:
            with zipfile.ZipFile(zip_path) as zip_file:
                pdf_members = [
                    name for name in zip_file.namelist()
                    if not name.endswith("/") and name.lower().endswith(".pdf")
                ]
                if not pdf_members:
                    logger.error("zip 中未找到 PDF 文件: %s", zip_path)
                    raise ValueError(f"zip 中未找到 PDF 文件: {zip_path}")
                if len(pdf_members) > 1:
                    logger.error("zip 中存在多个 PDF 文件，无法确定校验对象: %s", zip_path)
                    raise ValueError(f"zip 中存在多个 PDF 文件，无法确定校验对象: {zip_path}")

                with zip_file.open(pdf_members[0]) as pdf_file:
                    for chunk in iter(lambda: pdf_file.read(1024 * 1024), b""):
                        file_hash.update(chunk)
        except zipfile.BadZipFile as e:
            logger.error("zip 文件格式错误: %s", zip_path)
            raise ValueError(f"zip 文件格式错误: {zip_path}") from e

        md5 = file_hash.hexdigest()
        logger.info("zip 内 PDF 的 MD5 计算完成: %s", zip_path)
        return md5

    def VerifyChildrenAttachmentIntegrity(
        self,
        zip_path: str | Path,
        prop_path: str | Path,
    ) -> bool:
        """校验 zip 内 PDF 文件 MD5 是否与 prop 文件中的 hash 一致。"""
        zip_path = Path(zip_path)
        prop_path = Path(prop_path)
        logger.info("正在校验 Zotero 附件完整性，zip_path=%s，prop_path=%s", zip_path, prop_path)

        if not zip_path.is_file():
            logger.error("附件完整性校验失败: zip 文件不存在: %s", zip_path)
            raise FileNotFoundError(f"zip 文件不存在: {zip_path}")
        if not prop_path.is_file():
            logger.error("附件完整性校验失败: prop 文件不存在: %s", prop_path)
            raise FileNotFoundError(f"prop 文件不存在: {prop_path}")
        
        # 校验 hash
        try:
            root = ElementTree.parse(prop_path).getroot()
        except ElementTree.ParseError as e:
            logger.error("附件完整性校验失败: prop 文件格式错误: %s", prop_path)
            raise ValueError(f"prop 文件格式错误: {prop_path}") from e

        hash_text = root.findtext("hash")
        if not hash_text:
            logger.error("附件完整性校验失败: prop 文件中没有 hash 字段: %s", prop_path)
            raise ValueError(f"prop 文件中没有 hash 字段: {prop_path}")

        expected_hash = hash_text.strip().lower()
        actual_hash = self._CalculateZipPdfMD5(zip_path)
        if actual_hash != expected_hash:
            logger.error(
                "附件完整性校验失败: %s expected=%s, actual=%s",
                zip_path,
                expected_hash,
                actual_hash,
            )
            raise ValueError(
                f"附件完整性校验失败: {zip_path} "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        logger.info("附件完整性校验成功: %s", zip_path)
        return True

    def GetChildrenAttachment(
        self,
        ChildrenKey: str | int,
        FilePath: str | Path,
    ) -> dict[str, Path]:
        """从 WebDAV 下载并校验 Zotero 子附件文件。

        当用户需要获取 Zotero 中某个附件文件时使用这个工具，通常用于下载
        论文 PDF。调用前应先使用 ``GetParentsChildrenMetadatas`` 找到
        对应附件子条目的 key。本工具会下载该子附件在 WebDAV 中对应的
        ``.zip`` 和 ``.prop`` 文件；如果本地已存在同名文件则跳过下载；
        下载后会校验 zip 内 PDF 的 MD5 是否与 prop 文件记录的 hash 一致。

        Args:
            ChildrenKey: Zotero 子附件条目的 key。这个值应来自
                ``GetParentsChildrenMetadatas`` 返回结果中的子条目 ``key``，
                不要传入父条目的 key。
            FilePath: 本地保存目录，用于存放下载得到的 ``.zip`` 和
                ``.prop`` 文件。

        Returns:
            下载文件的本地路径字典，通常形如
            ``{"zip": Path(...), "prop": Path(...)}``。
        """
        children_key = str(ChildrenKey).strip()
        if not children_key:
            logger.error("Zotero 子附件下载失败: ChildrenKey 为空")
            raise ValueError("ChildrenKey 不为空！！")

        try:
            base_url = "zotero"
            save_dir = Path(FilePath)
            logger.info("正在下载 Zotero 子附件，ChildrenKey=%s，save_dir=%s", children_key, save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            attachment_paths = {}
            for suffix in ("zip", "prop"):
                attachment_paths[suffix] = self._DownloadWebDavFile(
                    remote_path=f"{base_url}/{children_key}.{suffix}",
                    local_path=save_dir / f"{children_key}.{suffix}",
                )
            
            logger.info("Zotero 子附件下载完成，ChildrenKey=%s", children_key)

            self.VerifyChildrenAttachmentIntegrity(
                zip_path=attachment_paths["zip"],
                prop_path=attachment_paths["prop"],
            )

            logger.info("Zotero 子附件校验完成，ChildrenKey=%s", children_key)
            return attachment_paths

        except Exception as e:
            logger.exception("Zotero 子附件下载或校验失败，ChildrenKey=%s", children_key)
            raise RuntimeError(f"下载或校验失败: {e}")


_zoteroclient: ZoteroClient | None = None
_zotero_client_tools: list[Callable[..., Any]] | None = None


def get_ZoteroClient() -> ZoteroClient:
    global _zoteroclient

    if _zoteroclient is None:
        _zoteroclient = ZoteroClient()

    return _zoteroclient


class LazyZoteroClient:
    def SearchOnParentTitle(
        self,
        title: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return get_ZoteroClient().SearchOnParentTitle(title=title, limit=limit)

    def GetParentMetadata(
        self,
        ParentKey: str | int,
    ) -> dict[str, Any]:
        return get_ZoteroClient().GetParentMetadata(ParentKey=ParentKey)

    def GetParentsChildrenMetadatas(
        self,
        ParentKey: str | int,
    ) -> list[dict[str, Any]]:
        return get_ZoteroClient().GetParentsChildrenMetadatas(ParentKey=ParentKey)

    def GetChildrenMetadata(
        self,
        ChildrenKey: str | int,
    ) -> dict[str, Any]:
        return get_ZoteroClient().GetChildrenMetadata(ChildrenKey=ChildrenKey)

    def VerifyChildrenAttachmentIntegrity(
        self,
        zip_path: str | Path,
        prop_path: str | Path,
    ) -> bool:
        return get_ZoteroClient().VerifyChildrenAttachmentIntegrity(
            zip_path=zip_path,
            prop_path=prop_path,
        )

    def GetChildrenAttachment(
        self,
        ChildrenKey: str | int,
        FilePath: str | Path,
    ) -> dict[str, Path]:
        return get_ZoteroClient().GetChildrenAttachment(
            ChildrenKey=ChildrenKey,
            FilePath=FilePath,
        )

    def __getattr__(self, name: str):
        return getattr(get_ZoteroClient(), name)


def get_ZoteroClientTools() -> list[Callable[..., Any]]:
    global _zotero_client_tools

    if _zotero_client_tools is None:
        client = get_ZoteroClient()
        _zotero_client_tools = [
            client.SearchOnParentTitle,
            client.GetParentsChildrenMetadatas,
            client.GetChildrenAttachment,
        ]

    return _zotero_client_tools


zoteroclient = LazyZoteroClient()
ZoteroClientTools = LazyToolList(get_ZoteroClientTools)


if __name__ == "__main__":
    zot = ZoteroClient()
    zot.GetChildrenAttachment("99IMFEBJ", FilePath="database/Cache")
    # zot.GetChildrenMetadata("99IMFEBJ")
    # files = zot.webdavclient.list("zotero")
    # print(files)

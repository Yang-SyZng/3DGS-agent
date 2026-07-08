import hashlib
import json
from pathlib import Path
from typing import Any, List, Sequence

from pymilvus import MilvusClient
from llama_index.core.schema import BaseNode
from llama_index.core.tools import FunctionTool

from .embedding import embedding

from config import setting
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MilvusVectorClient:
    def __init__(self,
        collection_name: str = None,
        collection_dim: int = None,
        db_directory: str = None,
    ):
        self.collection_name = collection_name or setting.Milvus_collection_name
        self.db_directory = db_directory or setting.Milvus_db_directory
        self.embedding_fn = embedding
        self.collection_dim = collection_dim or setting.EMBEDDING_DIM

        logger.info(f"初始化向量数据库...")

        self.milvusvector = MilvusClient(
            str(self.db_directory),
        )

        if not self.milvusvector.has_collection(self.collection_name):
            self.milvusvector.create_collection(
                collection_name=self.collection_name,
                dimension=self.collection_dim,
                id_type="string",
                max_length=128,
                metric_type="IP",
            )
        self.milvusvector.load_collection(self.collection_name)
        self._use_string_id = self._collection_uses_string_id()

    @property
    def client(self) -> MilvusClient:
        return self.milvusvector

    def __getattr__(self, name: str):
        return getattr(self.milvusvector, name)

    def _collection_uses_string_id(self) -> bool:
        description = self.milvusvector.describe_collection(self.collection_name)
        for field in description.get("fields", []):
            if field.get("name") != "id":
                continue
            field_type = str(field.get("type", "")).upper()
            return "VARCHAR" in field_type or "STRING" in field_type
        return False

    def _node_id_to_int(self, node_id: str) -> int:
        digest = hashlib.blake2b(node_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)

    def _node_to_row(self, node: BaseNode) -> dict[str, Any]:
        if node.embedding is None:
            raise ValueError(f"节点 {node.node_id} 缺少 embedding，请先调用 embedding.embed_nodes(nodes)")

        node_id = str(node.node_id)
        row_id: str | int = node_id if self._use_string_id else self._node_id_to_int(node_id)

        return {
            "id": node_id,
            "vector": node.embedding,
            # "node_id": node_id,
            "text": node.get_content(),
            "metadata": node.metadata,
            "node_json": json.dumps(node.to_dict(), ensure_ascii=False),
        }

    def add_documents(
        self,
        nodes: Sequence[BaseNode],
    ) -> list[Any]:
        if isinstance(nodes, BaseNode):
            nodes = [nodes]

        missing_embeddings = [node for node in nodes if node.embedding is None]
        if missing_embeddings:
            self.embedding_fn.embed_nodes(missing_embeddings)

        rows = [self._node_to_row(node) for node in tqdm(nodes, desc="processing")]
        if not rows:
            return []

        result = self.milvusvector.insert(
            collection_name=self.collection_name,
            data=rows,
        )
        ids = result.get("ids", [])
        logger.info(f"成功添加{len(ids)}个文档")
        return ids

    def search(self, query: str, top_k: int = 5):
        """在向量知识库中搜索与查询相关的文档。

        当 agent 需要检索项目知识、已索引文档，或查找与用户问题在语义上
        相似的上下文片段时使用此工具。

        Args:
            query: 自然语言搜索查询，用于描述需要检索的信息。
            top_k: 最多返回的相关结果数量，默认为 5。

        Returns:
            按向量相似度排序的 Milvus 搜索结果。每条结果可能包含匹配文档
            的 id、相似度分数、文本、元数据和序列化后的节点数据，具体字段
            取决于集合的输出配置。
        """
        vector = embedding.embed_query(query)
        res = self.milvusvector.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["id", "text"]
        )

        return res[0]


milvusvector = MilvusVectorClient()

ragtools = [
    FunctionTool.from_defaults(
        fn=milvusvector.search
        ),
]

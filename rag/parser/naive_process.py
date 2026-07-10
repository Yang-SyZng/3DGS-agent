import re
from typing import List
import hashlib

from llama_index.core import Document
from ...schema.paper_schema import SectionNode, SectionType, SECTION_KEYWORDS, CHILD_KEYWORDS

def load_markdown(path: str, metadata: dict | None = None) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return Document(
        text=text,
        metadata=metadata or {},
    )


def build_front_matter_node(front_matter: list[str], paper_id: str,) -> SectionNode | None:
    if not front_matter:
        return None

    front_text = "\n\n".join(front_matter).strip()
    if not front_text:
        return None

    node = SectionNode(
        title=front_matter[0],
        content=front_matter[1],
        level=1,
    )

    node.path = ["Paper Metadata"]
    node.section_id = generate_section_id(paper_id, node.path)

    node.semantic_type = SectionType.OTHER

    return node

def parse_nodes(data: Document, pattern: str = r'(?=^#{1,3}\s)') -> List[SectionNode]:
    split_1 = split_section(data, pattern)
    front_matter, split_2 = split_title(split_1)
    nodes = level_resolver(split_2)

    front_node = build_front_matter_node(front_matter, 'AbsGS')

    return front_node, nodes


def split_section(data: Document, pattern: str = r'(?=^#{1,3}\s)') -> List[str]:
    """
        1、切分Markdown
    """
    doc_text = data.text

    splited_sections = re.split(
        pattern,
        doc_text,
        flags=re.MULTILINE
    )
    splited = []
    for section in splited_sections:
        if section.strip():
            splited.append(section.strip())
    
    
    return splited

def split_title(splited_sections: List[List]) -> List[str | List[str]]:
    """
        2、切分标题
    """
    splited = []
    for section in splited_sections:
        splited.append(re.split(r'\n\n', section, maxsplit=1))
    # 头部块, 包含大标题，作者，以及额外的内容
    front_matter = splited[0]
    return front_matter, splited[1:]


def level_resolver(splited_sections: List[str | List[str]], windows: int = 1) -> List[SectionNode]:
    """
        3、level恢复：markdown -> 数字标题
    """

    nodes: List[SectionNode] = []
    for section in splited_sections:
        # markdown 判断
        # 判断开头 # 数量
        # markdown 尝试索取
        split_0 = re.split(r' ', section[0], maxsplit=1)
        if split_0[0].count('#') == 1:
            node = SectionNode(
                title=split_0[1],
                content=section[1] if len(section) > 1 else "",
                level=1
            )
            nodes.append(node)
        else: 
            has_number = bool(re.match(r'^\d+(\.\d+)*\s+', split_0[1]))
            if has_number:
                # 编号 尝试索取
                split_1 = re.split(r' ', split_0[1], maxsplit=1)
                point_count = split_1[0].count('.')
                node = SectionNode(
                    title=clean_heading_title(split_0[1]),
                    # 绝大多数标题总会带点内容，以防万一
                    content=section[1] if len(section) > 1 else "",
                    # 没有 一级标题 . 二级标题 ..三级标题 ....
                    level=point_count + 1
                )
                nodes.append(node)
            else:
                # 可能需要结合前面section的level来判断，绝大多数情况下是一级标题
                if len(nodes) > 0:
                    last_node = nodes[-1]
                    if last_node.level >= 0:
                        node = SectionNode(
                            title=split_0[1],
                            content=section[1] if len(section) > 1 else "",
                            level=1
                        )
                else:
                    node = SectionNode(
                        title=split_0[1],
                        content=section[1] if len(section) > 1 else "",
                        level=1
                        )
                nodes.append(node)
    return nodes

def build_tree(nodes: List[SectionNode], paper_id: str) -> List[SectionNode]:

    roots: List[SectionNode] = []
    stack: List[SectionNode] = []

    for node in nodes:
        if node.level == 1:
            node.path = [node.title]

            node.section_id = generate_section_id(paper_id, node.path)

            roots.append(node)
            stack = [node]
        
        else:
            # 找父节点
            while len(stack) >= node.level:
                stack.pop()
            
            parent = stack[-1]
            parent.children.append(node)

            node.path = (parent.path + [node.title])

            node.section_id = generate_section_id(paper_id, node.path)

            stack.append(node)

    return roots

def clean_heading_title(title):
    title = title.strip()

    # 阿拉伯数字
    title = re.sub(
        r"^\d+(?:\.\d+)*\.?\s+",
        "",
        title
    )

    # 罗马数字
    title = re.sub(
        r"^[IVX]+\.?\s+",
        "",
        title,
        flags=re.IGNORECASE
    )

    return title.strip()

def generate_section_id(paper_id: str, path: list[str]):
    raw = paper_id + "/" + "/".join(path)

    return hashlib.md5(raw.encode()).hexdigest()[:12]


def classify_section_tree(nodes: List[SectionNode]):
    for node in nodes:
        node.semantic_type = classify_title(node.title)
        assign_semantic_type(node, node.semantic_type)
    return nodes

def classify_title(title:str):
    title = clean_heading_title(title).lower()

    for section_type, keywords in SECTION_KEYWORDS.items():
        if any(keyword in title for keyword in keywords):
            return section_type
            
    return SectionType.OTHER

def assign_semantic_type(node: SectionNode, parent_type: SectionType):
    for child in node.children:
        child.semantic_type = classify_child(child, parent_type)
        assign_semantic_type(child, child.semantic_type)

def classify_child(node: SectionNode, parent_type: SectionType):
    title = clean_heading_title(node.title).lower()

    # 默认继承
    semantic_type = parent_type

    # 子章节覆盖规则
    for section_type, keywords in CHILD_KEYWORDS.items():
        if any(keyword in title for keyword in keywords):
            semantic_type = section_type
            break

    return semantic_type


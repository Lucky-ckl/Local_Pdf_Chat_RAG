"""
向量存储 —— FAISS 向量索引管理

学习要点：
- FAISS (Facebook AI Similarity Search) 是高效的向量相似度检索库
- IndexFlatL2: 暴力搜索，精确但慢。适合小数据集（<1万）
- IndexIVFFlat: 倒排索引，先聚类再搜索。适合中等数据集
- IndexIVFPQ: 乘积量化，牺牲精度换效率。适合大数据集（>10万）
- 本项目根据向量数量自动选择最优索引类型
"""

import logging
import os
import json
from datetime import datetime, timezone
import numpy as np
import faiss
from faiss import IndexFlatL2, IndexIVFFlat, IndexIVFPQ


# 持久化文件与 manifest 版本（P2.1，缺陷 D4：索引原为纯内存，重启即丢）。
FAISS_FILENAME = "faiss.index"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


class IndexPersistenceError(Exception):
    """索引持久化 / 加载相关错误的基类。"""


class IndexConfigMismatchError(IndexPersistenceError):
    """持久化索引的 manifest 与当前配置不一致；加载被显式拒绝，需重建索引。"""


class AutoFaissIndex:
    """
    自动选择 FAISS 索引类型的封装类

    根据数据量自动选择最优索引类型：
    - 小数据集（<1万）: FlatL2（精确搜索）
    - 中等数据集（1万-10万）: IVFFlat（近似搜索）
    - 大数据集（>10万）: IVFPQ（高效近似搜索）
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.nlist = None
        self.m = None
        self.nprobe = None
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000

    @property
    def ntotal(self):
        return self.index.ntotal if self.index else 0

    def select_index_type(self, num_vectors):
        """根据向量数量自动选择最优索引类型"""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = "FlatL2"
            self.index = IndexFlatL2(self.dimension)
            self.nprobe = 1
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = "IVFFlat"
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(10, max(1, int(self.nlist * 0.1)))
        else:
            self.index_type = "IVFPQ"
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8)
            self.nprobe = min(32, max(1, int(self.nlist * 0.05)))

        logging.info(f"选择索引类型: {self.index_type}，向量数: {num_vectors}")
        return self.index_type

    def train(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.train(vectors)

    def add(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"] and not self.index.is_trained:
            self.train(vectors)
        self.index.add(vectors)

    def search(self, query_vectors, k=5):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.nprobe = self.nprobe
        return self.index.search(query_vectors, k)

    def get_index_info(self):
        return {
            "index_type": self.index_type, "dimension": self.dimension,
            "nlist": self.nlist, "nprobe": self.nprobe, "size": self.ntotal
        }


class VectorStore:
    """
    向量存储管理器

    封装 FAISS 索引及其关联的文档内容和元数据映射。
    解决原代码中 4 个全局变量的管理问题。
    """

    def __init__(self):
        self.index = None           # AutoFaissIndex 实例
        self.contents_map = {}      # chunk_id -> 文本内容
        self.metadatas_map = {}     # chunk_id -> 元数据
        self.id_order = []          # 按顺序记录的 chunk_id 列表

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        构建 FAISS 索引

        Args:
            chunks: 文本片段列表
            chunk_ids: 片段 ID 列表
            metadatas: 元数据列表
            embeddings: 向量数组 (numpy, float32)
        """
        dimension = embeddings.shape[1]
        num_vectors = len(chunks)

        auto_index = AutoFaissIndex(dimension=dimension)
        auto_index.select_index_type(num_vectors)

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        auto_index.add(embeddings)
        self.index = auto_index
        logging.info(f"FAISS 索引构建完成，共 {self.index.ntotal} 个文本块，类型: {auto_index.index_type}")

    def add_index(self, chunks, chunk_ids, metadatas, embeddings):
        """把新 chunk **追加**到现有索引（增量入库：不清空、不重建已有向量）。

        与 ``build_index`` 的区别：保留已有的 contents / metadatas / id_order 与 FAISS 向量，
        只追加本次新增的 chunk，使多个文档可以累计检索（知识库目录模式）。
        索引为空时等价于 ``build_index``；已存在的 chunk_id 会被跳过，避免重复入库。
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        if not self.is_ready:
            self.build_index(chunks, chunk_ids, metadatas, embeddings)
            return
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("增量索引：向量数量与 chunk 数量不一致")
        if embeddings.shape[1] != self.index.dimension:
            raise ValueError(
                f"增量索引：向量维度 {embeddings.shape[1]} 与现有索引 {self.index.dimension} 不一致"
            )
        existing = set(self.contents_map)
        new_positions = [i for i, chunk_id in enumerate(chunk_ids) if chunk_id not in existing]
        if not new_positions:
            logging.info("增量索引：本次没有新增 chunk")
            return
        for i in new_positions:
            self.contents_map[chunk_ids[i]] = chunks[i]
            self.metadatas_map[chunk_ids[i]] = metadatas[i]
            self.id_order.append(chunk_ids[i])
        self.index.add(embeddings[new_positions])
        logging.info(f"增量索引追加完成，共 {self.index.ntotal} 个文本块")

    def search_with_scores(self, query_embedding, k=10):
        """
        搜索最相似的向量，并返回真实距离。

        索引为 L2 系列（IndexFlatL2 / IVFFlat / IVFPQ），返回的是**平方 L2 距离**，
        数值越小代表越相似；此处原样返回，不在本层归一化。

        Returns:
            (docs, doc_ids, metadatas, distances)
            distances 为 float 列表，与 docs / doc_ids / metadatas 一一对应。
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], [], []
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas, distances = [], [], [], []
            for position, faiss_idx in enumerate(I[0]):
                if faiss_idx != -1 and faiss_idx < len(self.id_order):
                    original_id = self.id_order[faiss_idx]
                    if original_id in self.contents_map:
                        docs.append(self.contents_map[original_id])
                        doc_ids.append(original_id)
                        metadatas.append(self.metadatas_map.get(original_id, {}))
                        distances.append(float(D[0][position]))
            return docs, doc_ids, metadatas, distances
        except Exception as e:
            logging.error(f"FAISS 检索错误: {str(e)}")
            return [], [], [], []

    def search(self, query_embedding, k=10):
        """
        搜索最相似的向量（保持既有三元组返回结构）。

        如需真实距离（用于混合检索分数归一化），请使用 ``search_with_scores``。

        Returns:
            (docs, doc_ids, metadatas)
        """
        docs, doc_ids, metadatas, _ = self.search_with_scores(query_embedding, k=k)
        return docs, doc_ids, metadatas

    @property
    def is_ready(self):
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_chunks(self):
        return self.index.ntotal if self.index is not None else 0

    def clear(self):
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        logging.info("向量存储已清空")

    def save(self, index_dir):
        """把 FAISS 索引与 chunk 元数据落盘到 ``index_dir``（P2.1）。

        - ``faiss.index``：FAISS 原生二进制（``faiss.write_index``）
        - ``chunks.json``：contents / metadatas / id_order 及索引类型参数
        - ``manifest.json``：embedding 模型名、chunk 参数、构建时间与块数（加载时用于一致性校验）

        返回是否写入成功；索引为空时不落盘。
        """
        if not self.is_ready:
            logging.warning("向量索引为空，跳过持久化")
            return False
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        from core.embeddings import EMBED_MODEL_NAME
        try:
            os.makedirs(index_dir, exist_ok=True)
            faiss.write_index(self.index.index, os.path.join(index_dir, FAISS_FILENAME))
            payload = {
                "version": 1,
                "contents": self.contents_map,
                "metadatas": self.metadatas_map,
                "id_order": self.id_order,
                "faiss_params": {
                    "index_type": self.index.index_type,
                    "dimension": self.index.dimension,
                    "nlist": self.index.nlist,
                    "m": self.index.m,
                    "nprobe": self.index.nprobe,
                },
            }
            with open(os.path.join(index_dir, CHUNKS_FILENAME), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            manifest = {
                "version": MANIFEST_VERSION,
                "embedding_model": EMBED_MODEL_NAME,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "num_chunks": self.total_chunks,
                "faiss_index_type": self.index.index_type,
                "built_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(os.path.join(index_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error("向量索引持久化失败: %s", e)
            return False
        logging.info(
            "FAISS 索引已持久化: %s（%d 个文本块）",
            os.path.join(index_dir, FAISS_FILENAME), self.total_chunks,
        )
        return True

    def load(self, index_dir):
        """从 ``index_dir`` 加载持久化索引（P2.1）。

        - 成功返回 ``True``；
        - 文件缺失 / 结构损坏 / 解析失败：清空并返回 ``False``（进入明确的空索引未就绪状态，
          不得伪装成加载成功）；
        - manifest 与当前配置不一致：抛 ``IndexConfigMismatchError``，显式拒绝加载过期索引。
        """
        self.clear()
        faiss_path = os.path.join(index_dir, FAISS_FILENAME)
        chunks_path = os.path.join(index_dir, CHUNKS_FILENAME)
        manifest_path = os.path.join(index_dir, MANIFEST_FILENAME)

        if not (os.path.exists(faiss_path) and os.path.exists(chunks_path)
                and os.path.exists(manifest_path)):
            logging.warning("未找到完整索引文件，进入空索引状态（索引未就绪）: %s", index_dir)
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logging.error("manifest.json 解析失败（索引损坏），进入空索引状态: %s", e)
            return False

        # 一致性校验：不匹配则显式拒绝（不静默加载过期索引）
        self._validate_manifest(manifest)

        try:
            faiss_index = faiss.read_index(faiss_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            params = payload.get("faiss_params", {})
            auto_index = AutoFaissIndex(dimension=params.get("dimension", faiss_index.d))
            auto_index.index = faiss_index
            auto_index.index_type = params.get("index_type")
            auto_index.nlist = params.get("nlist")
            auto_index.m = params.get("m")
            auto_index.nprobe = params.get("nprobe", 1)
            self.index = auto_index
            self.contents_map = payload["contents"]
            self.metadatas_map = payload["metadatas"]
            self.id_order = payload["id_order"]
        except Exception as e:
            logging.error("索引文件读取失败（索引损坏），进入空索引状态: %s", e)
            self.clear()
            return False

        logging.info("FAISS 索引已从磁盘加载: %d 个文本块", self.total_chunks)
        return True

    @staticmethod
    def _validate_manifest(manifest):
        """比对 manifest 的分块参数与 embedding 模型名与当前配置；不一致即抛异常。"""
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        from core.embeddings import EMBED_MODEL_NAME

        expected = {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBED_MODEL_NAME,
        }
        mismatches = []
        for key, current in expected.items():
            saved = manifest.get(key)
            if saved != current:
                mismatches.append(f"{key}: 索引={saved!r} 当前={current!r}")
        if mismatches:
            message = (
                "持久化索引与当前配置不匹配，已拒绝加载（请重建索引）: "
                + "; ".join(mismatches)
            )
            logging.error(message)
            raise IndexConfigMismatchError(message)


# 模块级单例
vector_store = VectorStore()


def save_persisted_index(index_dir=None):
    """保存 FAISS + BM25 + chunk 元数据 + manifest 到 ``INDEX_DIR``（P2.1）。

    顺序：先落 FAISS 索引 + chunk 元数据 + manifest（``vector_store.save``），再落 BM25
    （``bm25_manager.save``）。返回是否成功。
    """
    from config import INDEX_DIR
    from core.bm25_index import bm25_manager

    target = index_dir or INDEX_DIR
    if not vector_store.save(target):
        return False
    bm25_manager.save(target)
    return True


def load_persisted_index(index_dir=None):
    """启动时加载持久化索引，返回状态字典 ``{"status", "message", "chunks"}``。

    status 取值：
    - ``"loaded"``   ：FAISS + BM25 均成功加载；
    - ``"mismatch"`` ：manifest 与当前配置不一致，已显式拒绝加载（提示重建）；
    - ``"not_ready"``：文件缺失或损坏，进入空索引状态。
    """
    from config import INDEX_DIR
    from core.bm25_index import bm25_manager

    target = index_dir or INDEX_DIR
    try:
        ok = vector_store.load(target)
    except IndexConfigMismatchError as e:
        bm25_manager.clear()
        return {"status": "mismatch", "message": str(e), "chunks": 0}

    if not ok:
        bm25_manager.clear()
        return {"status": "not_ready", "message": "索引未就绪（文件缺失或损坏）", "chunks": 0}

    if not bm25_manager.load(target, contents_by_id=vector_store.contents_map):
        # 文件缺失或损坏导致 BM25 未能重建：整体进入未就绪，不得伪装成加载成功
        logging.error("BM25 索引加载失败，索引整体进入未就绪状态")
        vector_store.clear()
        bm25_manager.clear()
        return {"status": "not_ready", "message": "索引未就绪（BM25 索引缺失或损坏）", "chunks": 0}

    return {
        "status": "loaded",
        "message": f"索引已加载，共 {vector_store.total_chunks} 个文本块",
        "chunks": vector_store.total_chunks,
    }

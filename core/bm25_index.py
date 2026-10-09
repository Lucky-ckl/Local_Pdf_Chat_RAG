"""
BM25 稀疏检索索引 —— 基于关键词的传统检索

学习要点：
- BM25 (Best Matching 25) 是经典的信息检索算法
- 与向量语义检索互补：语义检索擅长理解意图，BM25 擅长精确关键词匹配
- 中文需要先分词（jieba），英文可直接按空格分
- 两者混合使用（Hybrid Search）可以显著提升检索效果
"""

import logging
import json
import os
import numpy as np
import jieba
from rank_bm25 import BM25Okapi


# 持久化文件名（P2.1）。BM25 只持久化 ids + 分词语料，启动时重建 BM25Okapi 对象，
# 不对 BM25Okapi 实例做 pickle（WBS v1.0 P2.1）。
BM25_FILENAME = "bm25.json"
BM25_PERSIST_VERSION = 1


class BM25IndexManager:
    """
    BM25 检索索引管理器

    负责构建、搜索和管理 BM25 索引。
    使用 jieba 分词以支持中文检索。
    """

    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    def build_index(self, documents, doc_ids):
        """构建 BM25 索引"""
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logging.info(f"BM25 索引构建完成，共索引 {len(documents)} 个文档")
        return True

    def search(self, query, top_k=5):
        """使用 BM25 检索相关文档"""
        if not self.bm25_index:
            return []

        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                results.append({
                    'id': self.doc_mapping[idx],
                    'score': float(bm25_scores[idx]),
                    'content': self.raw_corpus[idx]
                })
        return results

    def clear(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    def save(self, index_dir):
        """把 BM25 索引持久化到 ``index_dir/bm25.json``。

        只写入 chunk id 顺序与分词语料（``ids`` + ``tokenized``），**不序列化 BM25Okapi
        对象**；加载时用同样的语料重建，保证跨进程可复现。返回是否写入成功。
        """
        if not self.bm25_index or not self.tokenized_corpus:
            logging.warning("BM25 索引为空，跳过持久化")
            return False

        ids = [self.doc_mapping[i] for i in sorted(self.doc_mapping)]
        payload = {
            "version": BM25_PERSIST_VERSION,
            "ids": ids,
            "tokenized": self.tokenized_corpus,
        }
        try:
            os.makedirs(index_dir, exist_ok=True)
            path = os.path.join(index_dir, BM25_FILENAME)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            logging.error("BM25 索引持久化失败: %s", e)
            return False

        logging.info("BM25 索引已持久化: %s（%d 篇）", path, len(ids))
        return True

    def load(self, index_dir, contents_by_id=None):
        """从 ``index_dir/bm25.json`` 重建 BM25 索引。

        缺失、损坏或结构不一致时**不抛异常**，而是清空索引并返回 ``False``（调用方据此
        进入空索引/未就绪状态，不得伪装成加载成功）。

        ``contents_by_id``：可选的 ``{chunk_id: 原文}`` 映射，用于恢复检索结果的 content 字段
        （BM25 自身只持久化 ids + 分词语料，原文由 chunk 存储提供）。
        """
        self.clear()
        path = os.path.join(index_dir, BM25_FILENAME)
        if not os.path.exists(path):
            logging.warning("BM25 持久化文件不存在: %s", path)
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ids = payload["ids"]
            tokenized = payload["tokenized"]
            if not isinstance(ids, list) or not isinstance(tokenized, list):
                raise ValueError("ids/tokenized 类型异常")
            if len(ids) != len(tokenized):
                raise ValueError("ids 与 tokenized 数量不一致")
            if not ids:
                raise ValueError("空语料")
        except Exception as e:
            logging.error("BM25 索引加载失败（%s），进入空索引状态: %s", type(e).__name__, e)
            self.clear()
            return False

        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(ids)}
        self.tokenized_corpus = [list(tokens) for tokens in tokenized]
        if contents_by_id is not None:
            self.raw_corpus = [contents_by_id.get(doc_id, "") for doc_id in ids]
        else:
            self.raw_corpus = ["" for _ in ids]
        try:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
        except Exception as e:
            logging.error("BM25 索引重建失败（%s），进入空索引状态: %s", type(e).__name__, e)
            self.clear()
            return False

        logging.info("BM25 索引已从磁盘重建: %s（%d 篇）", path, len(ids))
        return True


# 模块级单例
bm25_manager = BM25IndexManager()

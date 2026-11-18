# rag-eval

对中文技术文档比较分块与混合检索的轻量评测台。Python 3.10+，默认无网络、无需密钥；运行时只使用标准库。

## 数据来源与许可

语料来自 Vue 官方中文文档 [vuejs-translations/docs-zh-cn](https://github.com/vuejs-translations/docs-zh-cn)，以 CC BY 4.0 许可提供；署名：Yuxi (Evan) You and Vue documentation contributors。项目代码使用 MIT 许可；数据保留其原有许可，评测题由数据提供方人工撰写。数据与语料已随仓库提供：`data/`（评测题与文档元信息）、`corpus/docs-zh-cn/src/`（语料原文）与 `corpus/docs-zh-cn/LICENSE`（CC BY 4.0 许可文本）。运行时只读取这些输入，不写入。

## 评分规则原文

> 对每道题，看召回的 Top-K 块——标准答案小节的内容**被单个块完整覆盖**记 2 分；**需要两个块合并才覆盖**记 1 分；否则 0 分。判定用字符串包含 + 归一化（去空白与标点）后的子串匹配，覆盖阈值取 `gold_excerpt` 的 80%。输出满分 100 的总分、命中率、以及按 `cat` 和 `has_code_block` 分组的明细。

实际判定要求块属于黄金文档和标题路径；摘录归一化后按顺序做最长片段匹配（单片段最长 12 字），覆盖比例达到 80% 即达阈值，允许非连续覆盖。两块合并时把两块的归一化文本顺序拼接后整体匹配，并尝试去重叠的拼接方式。黄金摘录中的 `[代码块]` 在评分时对应完整代码块标记；代码原文仍保存在块中。命中率是得 1 分或 2 分的题目比例，满分率是得 2 分的题目比例。

## 复现

仓库自包含：语料与评测集都在仓库内，clone 后无需额外准备：

```text
git clone https://github.com/<your-name>/rag-eval.git
cd rag-eval
python run_eval.py
```

运行测试（pytest 为唯一可选依赖）：

```sh
python run_eval.py
python -m pip install pytest
pytest tests
```

产物为 `out/docs.jsonl`、各策略的 SQLite 文件、`out/results.json`、`out/report.md` 和可直接打开的 `out/report.html`。默认评测向量/BM25 权重为 1/0、0.5/0.5、0/1，Top-K 为 5。网页内可输入任意问题，选策略和权重查看检索分数及命中块；页面使用离线 mock 和字 bi-gram，即使评测启用了其他可选模型也是如此。

自定义配置示例：`python run_eval.py --top-k 5 --vector-weight 0.5 --bm25-weight 0.5`。可安装 jieba 并传 `--jieba`，未安装时自动回退字 bi-gram。可选 `--embedder openai-compatible` 使用环境变量 `BASE_URL`、`API_KEY`、`MODEL`；该模式需要可访问的服务。可选 `--rerank cross-encoder` 尝试本地已缓存模型，失败时保持粗排并在结果中标记 `rerank_applied=false`，不会下载模型。

`fixed` 默认每块 500 字符，`semantic` 按句子和词面重合度合并，`recursive` 优先标题、小节、段落、句子再硬切，默认长度 800 和重叠 150。为保证代码完整，代码块作为不可拆单元，可能超过长度限制；递归块首还会附加标题路径。各策略保持小节边界，避免错误标注标题。重叠以不拆代码和不超过目标长度为前提，因此边界处可能少于指定值。

## 已知局限

语料来自单一领域，评测集仅 50 条；mock 嵌入是可复现的字符二元组哈希，不代表真实语义模型的效果。Markdown 解析面向本语料，不是完整 CommonMark 实现。评分只统计摘录的顺序覆盖率（阈值 80%），且依赖黄金文档、标题与占位符的质量，不能替代人工判断。页面展示检索分数，评测分数见报告。

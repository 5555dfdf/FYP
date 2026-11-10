# 本地文献证据

来源核查日期：2026-11-09。第三方全文、检索响应、提取文本和渲染图按仓库既有规则仅保存在本地，不提交。可分享的分析见上一级 `review_memo_zh.md`，来源 URL、SHA256、响应与失败记录见 `source_manifest.json` 和 `fetch_round*.json`。

关键文件：

- `arxiv_html.txt` / `arxiv_html_clean.txt`：REHAB26 原始论文官方 HTML 与提取文本。
- `rehab26_pdf.pdf`：arXiv:2609.35726v2 官方 PDF；原始字节 SHA256 见清单。
- `rehab26_page_*.txt`：PDF 逐页提取文本。
- `features-07.png` / `table3-17.png`：表 1 与表 3 的实际页面渲染，均已视觉核对。
- `zenodo_live.txt`：官方数据集版本记录。
- `openalex_exact.txt` / `openalex_rehab26_broad.txt` / `openalex_cites_*.json`：同数据集检索和前向引用结果。
- `personalized_springer.txt`：2024 年前代个体化方法的官方摘要页，非全文。
- `fine_tuning_elsevier_api.txt`：2025 年姿态微调论文的官方核心元数据，非全文。
- `rehab24_pose_finetuning_2025.pdf`：用户补充的上述论文出版商全文，原始文件与副本 SHA256 相同；全文补充核查见上一级 `rehab24_pose_finetuning_fulltext_zh.md`。
- `pose_finetuning_full.txt` / `pose_finetuning_page_*.txt`：该论文的全文与逐页提取文本；`pose-results-09.png` / `pose-results-10.png` 为已视觉核对的结果页。
- `mobile_openalex_record.txt`：2026 年移动演示论文的索引摘要与书目。
- `new_pose_html.txt`：Fit3D 论文全文，用于排除搜索中出现的不同数据集候选。

重新采集：在 `conda activate fyp` 环境执行 `python -X utf8 literature/2026-11-09_rehab26_methods/collect_sources.py --fetch`。默认不带 `--fetch` 只显示来源；新采集会进入单独时间戳目录，不覆盖已核对原件。访问出版商页面成功不表示获得全文，重新阅读时仍须判断响应内容。

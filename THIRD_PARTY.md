# Code and Data Sources

The repository preserves the original nanoGPT MIT licence in `LICENSE`, including Andrej Karpathy's attribution.
The MQAR generator is from [Zoology](https://github.com/HazyResearch/zoology/tree/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb), revision `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, under the licence retained in `zoology/LICENSE.md`.
This distribution removes Zoology's configuration and logging framework and supplies a minimal `DataSegment` container; the `multiquery_ar` generation function is unchanged.

PyTorch, Triton, Transformers, datasets, tiktoken and the evaluation harness are installed as dependencies rather than copied into this repository.
The Qwen GQA code is sourced from the project's `prior-corpus-experiments` revision `9b9dc79`.
It is isolated from the native-model implementation because the recorded experiments used different kernel revisions.

## External Assets

Download data and third-party weights from their original providers and follow their terms.
The source-code licence does not grant rights to the underlying datasets or to separately released weights.

| Asset | Upstream source |
| --- | --- |
| FineWeb-Edu | [Dataset card and licensing information](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#licensing-information) |
| SST-2 | [GLUE](https://huggingface.co/datasets/nyu-mll/glue) and [original dataset card](https://huggingface.co/datasets/stanfordnlp/sst2) |
| BoolQ | [Boolean Questions](https://github.com/google-research-datasets/boolean-questions) |
| QuALITY | [Dataset and article-level licences](https://github.com/nyu-mll/quality) |
| HellaSwag | [Original repository](https://github.com/rowanz/hellaswag) |
| PIQA | [Original dataset](https://github.com/ybisk/ybisk.github.io/tree/master/piqa) |
| ARC-Easy | [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) |
| Qwen3-4B | [Model card](https://huggingface.co/Qwen/Qwen3-4B) |
| OpenWebMath | [Dataset card](https://huggingface.co/datasets/open-web-math/open-web-math) |
| The Stack Smol | [Dataset card and access conditions](https://huggingface.co/datasets/bigcode/the-stack-smol) |

FineWeb-Edu is distributed under ODC-By 1.0; the rights in the underlying web pages and Common Crawl's terms remain relevant, as explained in its dataset card.
Do not redistribute benchmark text, downloaded corpora or model caches with the source release.
The authors must confirm the licence for their new contributions and trained weights before public publication; the retained upstream licence is not a substitute for that decision.

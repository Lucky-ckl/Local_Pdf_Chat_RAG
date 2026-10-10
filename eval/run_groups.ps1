# P5.4 四组「同集同参」对比驱动脚本（WBS v1.0 §8 P5.4）
# 用法： pwsh -File eval\run_groups.ps1 [-Tag <时间戳>] [-BaselineDir <旧代码目录>]
param(
    [string]$Tag = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    [string]$BaselineDir = "$env:TEMP\rag_p5_baseline"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$pych = Join-Path $repo '.venv\Scripts\python.exe'
$evalSet = Join-Path $repo 'eval\eval_set.json'
$corpus = Join-Path $repo 'eval\corpus'

Write-Host "run tag: $Tag"

function Invoke-Group {
    param([string]$Group, [string]$Script, [string]$IndexDir, [string[]]$Extra)
    $outDir = Join-Path $repo "eval\results\${Tag}_$Group"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $args = @($Script, '--group', $Group, '--eval-set', $evalSet, '--corpus', $corpus,
              '--index-dir', $IndexDir, '--out', (Join-Path $outDir 'retrieval.json')) + $Extra
    Write-Host "== group $Group =="
    & $pych @args *>&1 | Tee-Object -FilePath (Join-Path $outDir 'run.log')
}

# ④ 全量改造：重排开启（本地缓存 cross-encoder）+ OCR 开启
$env:RERANK_METHOD = 'cross_encoder'
$env:RERANK_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
$env:OCR_ENABLED = 'true'
Invoke-Group -Group 'full' -Script 'eval\run_eval.py' -IndexDir 'eval\index\full'

# ② 仅修重排（P1）：重排开启 + OCR 关闭
$env:OCR_ENABLED = 'false'
Invoke-Group -Group 'rerank_only' -Script 'eval\run_eval.py' -IndexDir 'eval\index\rerank_only'

# ③ 仅加 OCR（P2）：重排关闭（RERANK_METHOD=none → 真实 RRF）+ OCR 开启
$env:RERANK_METHOD = 'none'
$env:OCR_ENABLED = 'true'
Invoke-Group -Group 'ocr_only' -Script 'eval\run_eval.py' -IndexDir 'eval\index\ocr_only'

# ① 改造前基线：38b39f6 导出的旧代码，独立入库（无 OCR、重排为旧实现）
$baseEval = Join-Path $BaselineDir 'eval'
New-Item -ItemType Directory -Force -Path $baseEval | Out-Null
Copy-Item (Join-Path $repo 'eval\run_eval.py') (Join-Path $baseEval 'run_eval.py') -Force
Invoke-Group -Group 'baseline' -Script (Join-Path $baseEval 'run_eval.py') `
    -IndexDir (Join-Path $BaselineDir 'eval\index\baseline')

Write-Host "done: eval\results\${Tag}_*"

# ==============================================================================
# vLLM Local Serving Script for Windows PowerShell
# Target Model: Qwen/Qwen2.5-3B-Instruct or Qwen/Qwen3.5-2B
# ==============================================================================

param (
    [string]$ModelName = "Qwen/Qwen2.5-3B-Instruct",
    [int]$Port = 8000,
    [string]$HostIP = "0.0.0.0"
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Starting local vLLM OpenAI-Compatible Inference Server" -ForegroundColor Green
Write-Host " Model: $ModelName" -ForegroundColor Yellow
Write-Host " Endpoint: http://${HostIP}:${Port}/v1" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

# Test if vLLM module is installed
$vllmInstalled = python -c "import vllm; print('OK')" 2>$null
if ($vllmInstalled -ne "OK") {
    Write-Host "[NOTICE] vLLM package is not detected in current Python environment." -ForegroundColor Yellow
    Write-Host "Tip: If you have Ollama running with 'qwen3:1.7b' at http://localhost:11434," -ForegroundColor Green
    Write-Host "the Telco Workbench automatically detects and uses it without needing vLLM!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To install vLLM on Windows/WSL2: pip install vllm" -ForegroundColor White
    exit
}

# Run vLLM API server
python -m vllm.entrypoints.openai.api_server `
    --model $ModelName `
    --host $HostIP `
    --port $Port `
    --gpu-memory-utilization 0.90 `
    --max-model-len 4096 `
    --dtype float16 `
    --tensor-parallel-size 1 `
    --trust-remote-code

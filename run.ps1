Set-Location $PSScriptRoot
Set-Location "$PSScriptRoot\project"
& "$PSScriptRoot\customer_churn\Scripts\python.exe" -m uvicorn app.main:app --port 8000

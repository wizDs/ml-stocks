cd C:\local_repos\stocks\source
explorer 'http://127.0.0.1:8000/rf/GN.CO?days=5'
uvicorn api:app --reload
pause
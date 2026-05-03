"""
ContextOS — Entry Point
Run with: uvicorn main:app --reload
"""

from context_os.api.routes import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

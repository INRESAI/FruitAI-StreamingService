if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app="api:app", host="0.0.0.0", port=8000, debug=True, reload=True)
    uvicorn.run(app="api:app", host="0.0.0.0", port=4600)

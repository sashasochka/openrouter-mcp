import importlib
modules = ['pydantic','fastmcp','httpx','PIL']
for m in modules:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, '__version__', None)
        print(f"{m}: OK, version={ver}")
    except Exception as e:
        print(f"{m}: ERROR: {type(e).__name__}: {e}")

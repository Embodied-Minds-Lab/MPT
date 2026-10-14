def setup_jar(extra_dir: str):
    """
    Configure pyjnius classpath to include Anserini fatjar and additional Lucene jars
    BEFORE importing pyserini or pyjnius.
    """
    import jnius_config
    import pyserini.setup as setup
    import os,glob

    jar_dir = os.path.join(os.path.dirname(setup.__file__), "resources", "jars")
    cands = glob.glob(os.path.join(jar_dir, "anserini-*-fatjar.jar"))
    if not cands:
        raise RuntimeError(f"No anserini fatjar under: {jar_dir}")
    fatjar = max(cands, key=os.path.getmtime)

    extra = [
        os.path.join(extra_dir, "lucene-highlighter-9.9.1.jar"),
        os.path.join(extra_dir, "lucene-queries-9.9.1.jar"),
        os.path.join(extra_dir, "lucene-memory-9.9.1.jar"),
    ]
    for p in [fatjar, *extra]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    jnius_config.add_options("-Djava.awt.headless=true")
    jnius_config.set_classpath(fatjar, *extra)

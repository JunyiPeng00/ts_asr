try:
    from wesep.cli.extractor import load_model  # noqa
    from wesep.cli.extractor import load_model_local  # noqa
except ModuleNotFoundError:
    # Allow lightweight subpackages such as dataset/modules to be imported
    # without requiring the full runtime extraction dependency stack.
    load_model = None  # type: ignore[assignment]
    load_model_local = None  # type: ignore[assignment]

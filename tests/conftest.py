import pytest


# Needed so pytest-asyncio doesn't require @pytest.mark.asyncio on every test
def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            if hasattr(item, "function") and hasattr(item.function, "__wrapped__"):
                item.add_marker(pytest.mark.asyncio)

"""Theme/stylesheet tests.

Icon generation needs a QApplication (QPixmap can't be constructed without one), so
these run under the offscreen platform like the rest of the UI suite.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from guitar_trainer.ui import theme  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def isolated_icon_dir(tmp_path, monkeypatch):
    """Redirect icon caching to a temp dir so tests never touch the real
    ~/.local/share/guitar-trainer/icons — see AGENTS.md's data-pollution gotcha."""
    monkeypatch.setattr(theme, "_icon_dir", lambda: tmp_path)
    return tmp_path


class TestArrowIcons:
    def test_generates_every_expected_icon(self, qapp, isolated_icon_dir):
        paths = theme._ensure_arrow_icons()
        assert set(paths) == set(theme._ARROW_SPECS)
        for path in paths.values():
            assert path.exists()
            assert path.stat().st_size > 0

    def test_icons_are_cached_not_regenerated(self, qapp, isolated_icon_dir):
        first = theme._ensure_arrow_icons()
        mtimes = {name: path.stat().st_mtime_ns for name, path in first.items()}

        second = theme._ensure_arrow_icons()
        for name, path in second.items():
            assert path.stat().st_mtime_ns == mtimes[name]

    def test_disabled_variants_are_a_dimmer_colour(self, qapp, isolated_icon_dir):
        _, _, _, enabled_color = theme._ARROW_SPECS["spin-up-arrow.png"]
        _, _, _, disabled_color = theme._ARROW_SPECS["spin-up-arrow-disabled.png"]
        assert disabled_color != enabled_color

    def test_icon_dir_is_created(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.core.config as config_module

        target = tmp_path / "nested"
        monkeypatch.setattr(config_module, "data_dir", lambda: target)
        result = theme._icon_dir()
        assert result.is_dir()
        assert result == target / "icons"


class TestStylesheet:
    def test_returns_nonempty_css(self, qapp, isolated_icon_dir):
        css = theme.stylesheet()
        assert "QWidget" in css
        assert "QComboBox" in css

    def test_references_the_generated_icon_paths(self, qapp, isolated_icon_dir):
        css = theme.stylesheet()
        icons = theme._ensure_arrow_icons()
        for path in icons.values():
            assert path.as_posix() in css

    def test_does_not_set_a_global_font_size(self, qapp, isolated_icon_dir):
        """Regression guard: a stylesheet-level font-size silently overrides
        setFont() on every widget, including the large note/prompt readouts."""
        css = theme.stylesheet()
        assert "font-size" not in css

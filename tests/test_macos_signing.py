"""Developer ID + notarization helpers for the macOS .pkg (no codesign I/O)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGNING_PATH = ROOT / "packaging" / "macos_signing.py"

SAMPLE_DEV_ONLY = """\
  1) AAA000 "Apple Development: dev@example.com (TEAMID)"
     1 valid identities found
"""

SAMPLE_DEVELOPER_ID = """\
  1) AAA111 "Developer ID Application: Jane Doe (TEAMID)"
  2) BBB222 "Developer ID Installer: Jane Doe (TEAMID)"
  3) CCC333 "Apple Development: foo@bar.com (XYZ)"
     3 valid identities found
"""


def _load_signing():
    spec = importlib.util.spec_from_file_location("audiotools_macos_signing", SIGNING_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_macos_signing_module_exists():
    assert SIGNING_PATH.is_file()


def test_rejects_apple_development_for_distribution():
    signing = _load_signing()
    ids = signing.parse_identities(SAMPLE_DEV_ONLY)
    assert ids.application is None
    assert ids.installer is None
    assert ids.can_distribute() is False


def test_selects_developer_id_pair_and_ignores_development_cert():
    signing = _load_signing()
    ids = signing.parse_identities(SAMPLE_DEVELOPER_ID)
    assert ids.application == "Developer ID Application: Jane Doe (TEAMID)"
    assert ids.installer == "Developer ID Installer: Jane Doe (TEAMID)"
    assert ids.can_distribute() is True


def test_codesign_commands_use_hardened_runtime_timestamp_and_entitlements(tmp_path):
    signing = _load_signing()
    app = tmp_path / "AudioTools.app"
    macho = app / "Contents" / "MacOS" / "AudioTools"
    entitlements = ROOT / "packaging" / "entitlements.plist"
    cmds = signing.codesign_app_commands(
        app,
        "Developer ID Application: Jane Doe (TEAMID)",
        entitlements,
        macho_files=[macho],
    )
    joined = " ".join(part for cmd in cmds for part in cmd)
    assert "--options" in joined
    assert "runtime" in joined
    assert "--timestamp" in joined
    assert "--entitlements" in joined
    assert str(entitlements) in joined
    assert cmds[-1][-1] == str(app)


def test_productsign_command_uses_developer_id_installer(tmp_path):
    signing = _load_signing()
    src = tmp_path / "in.pkg"
    dest = tmp_path / "out.pkg"
    cmd = signing.productsign_command(
        src, dest, "Developer ID Installer: Jane Doe (TEAMID)"
    )
    assert cmd[0] == "productsign"
    assert "--sign" in cmd
    assert "Developer ID Installer: Jane Doe (TEAMID)" in cmd
    assert str(src) in cmd
    assert str(dest) in cmd


def test_notarize_commands_submit_wait_and_staple(tmp_path):
    signing = _load_signing()
    pkg = tmp_path / "AudioTools.pkg"
    cmds = signing.notarize_commands(pkg, keychain_profile="audio-tools-notary")
    submit = cmds[0]
    staple = cmds[1]
    assert "notarytool" in submit
    assert "submit" in submit
    assert "--wait" in submit
    assert "--keychain-profile" in submit
    assert "audio-tools-notary" in submit
    assert "stapler" in staple
    assert "staple" in staple
    assert str(pkg) in staple


def test_entitlements_allow_pyinstaller_hardened_runtime():
    text = (ROOT / "packaging" / "entitlements.plist").read_text(encoding="utf-8")
    assert "com.apple.security.cs.allow-jit" in text
    assert "com.apple.security.cs.allow-unsigned-executable-memory" in text
    assert "com.apple.security.cs.disable-library-validation" in text


def test_make_pkg_signs_and_defaults_to_desktop_version():
    script = (ROOT / "packaging" / "make_pkg.sh").read_text(encoding="utf-8")
    assert "macos_signing.py" in script
    assert "AUDIO_TOOLS_VERSION:-0.1.6" in script
    assert "pkgbuild" in script
    assert "macos-arm64-silicon.pkg" in script
    assert "macos-x64-intel.pkg" in script
    assert "--component-plist" in script
    assert "pkg_component.plist" in script


def test_make_app_copies_without_quarantine_xattrs():
    script = (ROOT / "packaging" / "make_app.sh").read_text(encoding="utf-8")
    assert "AUDIO_TOOLS_VERSION:-0.1.6" in script
    assert "--noqtn" in script or "xattr -cr" in script


def test_pkg_component_is_not_relocatable():
    """Installer must write /Applications, not upgrade a leftover dist/ copy."""
    text = (ROOT / "packaging" / "pkg_component.plist").read_text(encoding="utf-8")
    assert "<key>BundleIsRelocatable</key>" in text
    assert "<false/>" in text
    assert "<key>BundleIsVersionChecked</key>" in text
    assert "AudioTools.app" in text


def test_postinstall_requires_applications_bundle():
    text = (ROOT / "packaging" / "pkg_scripts" / "postinstall").read_text(encoding="utf-8")
    assert "/Applications/AudioTools.app" in text
    assert "was not installed" in text


def test_postinstall_launches_app_as_console_user():
    text = (ROOT / "packaging" / "pkg_scripts" / "postinstall").read_text(encoding="utf-8")
    assert "/dev/console" in text
    assert "launchctl asuser" in text
    assert "open -n" in text
    assert "loginwindow" in text


def test_gitignore_excludes_local_build_python():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".build/" in text
    assert "dogfood-output/" in text

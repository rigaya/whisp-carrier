from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Iterable


ROCM_DLL_IMPORTS = frozenset({"hipblas.dll", "amdhip64_7.dll"})


def is_rocm_imports(import_names: Iterable[str]) -> bool:
    """Return whether the PE imports identify a ROCm build."""
    normalized = {name.lower() for name in import_names}
    return ROCM_DLL_IMPORTS <= normalized


def pe_import_names(dll_path: Path) -> set[str]:
    """Read imported DLL names from a PE file."""
    import pefile

    pe = pefile.PE(str(dll_path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        )
        return {
            entry.dll.decode("ascii", errors="replace").lower()
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", ())
        }
    finally:
        pe.close()


def ctranslate2_is_rocm(dll_path: Path) -> bool:
    """Identify a ROCm CTranslate2 DLL from its PE imports."""
    return is_rocm_imports(pe_import_names(dll_path))


def elf_needed_names(shared_object: Path) -> set[str]:
    """Read DT_NEEDED entries from an ELF shared library."""
    from elftools.elf.elffile import ELFFile

    with shared_object.open("rb") as stream:
        elf = ELFFile(stream)
        return {
            tag.needed.lower()
            for segment in elf.iter_segments()
            if segment.header.p_type == "PT_DYNAMIC"
            for tag in segment.iter_tags()
            if tag.entry.d_tag == "DT_NEEDED"
        }


def ctranslate2_linux_is_rocm(
    package_dir: Path, maps_path: Path = Path("/proc/self/maps")
) -> bool:
    """Identify a ROCm CTranslate2 installation from its ELF dependencies."""
    candidates = list(package_dir.glob("_ext*.so"))
    try:
        distribution = metadata.distribution("ctranslate2")
        candidates.extend(
            Path(distribution.locate_file(item))
            for item in distribution.files or ()
            if item.name.startswith("libctranslate2") and ".so" in item.name
        )
    except metadata.PackageNotFoundError:
        pass

    # Also inspect loaded CTranslate2 libraries from source installations.
    if maps_path.is_file():
        for line in maps_path.read_text(encoding="utf-8").splitlines():
            path = line.split(maxsplit=5)[-1]
            if Path(path).name.startswith("libctranslate2") and ".so" in path:
                candidates.append(Path(path))

    inspected = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved in inspected:
            continue
        inspected.add(resolved)
        needed = elf_needed_names(resolved)
        if any(name.startswith(("libhipblas.so", "libamdhip64.so", "libhiprand.so"))
               for name in needed):
            return True
    if not inspected:
        raise RuntimeError("Could not find a Linux CTranslate2 shared library")
    return False

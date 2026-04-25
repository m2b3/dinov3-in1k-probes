"""HuggingFace repo ID construction for DINOv3 backbones and probes."""

VARIANTS = ("vits16", "vits16plus", "vitb16", "vitl16", "vith16plus")

HF_OWNER = "canvit"


def dinov3_backbone_repo(variant: str) -> str:
    """'vits16plus' → 'facebook/dinov3-vits16plus-pretrain-lvd1689m'."""
    assert variant in VARIANTS, f"Unknown variant {variant!r}, expected one of {VARIANTS}"
    return f"facebook/dinov3-{variant}-pretrain-lvd1689m"


def probe_repo(variant: str, image_size: int = 512, *, owner: str = HF_OWNER) -> str:
    """'vits16plus' → 'canvit/dinov3-vits16plus-lvd1689m-in1k-512x512-linear-clf-probe'."""
    assert variant in VARIANTS, f"Unknown variant {variant!r}, expected one of {VARIANTS}"
    return f"{owner}/dinov3-{variant}-lvd1689m-in1k-{image_size}x{image_size}-linear-clf-probe"


def dinov3_repo_from_model_name(model_name: str) -> str:
    """'dinov3_vits16plus' → 'facebook/dinov3-vits16plus-pretrain-lvd1689m'."""
    slug = model_name.replace("_", "-")
    return f"facebook/{slug}-pretrain-lvd1689m"


def model_name_from_repo(repo: str) -> str:
    """'facebook/dinov3-vitb16-pretrain-lvd1689m' → 'dinov3_vitb16'."""
    slug = repo.split("/")[-1]
    assert "-pretrain" in slug, f"Expected '-pretrain' in repo slug: {repo}"
    name = "_".join(slug.split("-pretrain")[0].split("-"))
    assert name.startswith("dinov3_"), f"Parsed name {name!r} doesn't look like a DINOv3 model (from {repo})"
    return name

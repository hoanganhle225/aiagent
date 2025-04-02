import minerl
import os
os.environ["MINERL_DATA_ROOT"] = "D:/Git/hoanganhle225/aiagent/aiagent-python/downloads"

ENVIRONMENTS = [
    'MineRLTreechop-v0',
    'MineRLNavigate-v0',
    'MineRLNavigateDense-v0',
    'MineRLNavigateExtreme-v0',
    'MineRLNavigateExtremeDense-v0',
    'MineRLObtainDiamond-v0',
    'MineRLObtainDiamondDense-v0',
    'MineRLObtainIronPickaxe-v0',
    'MineRLObtainIronPickaxeDense-v0',

    'MineRLTreechopVectorObf-v0',
    'MineRLNavigateVectorObf-v0',
    'MineRLNavigateDenseVectorObf-v0',
    'MineRLNavigateExtremeVectorObf-v0',
    'MineRLNavigateExtremeDenseVectorObf-v0',
    'MineRLObtainDiamondVectorObf-v0',
    'MineRLObtainDiamondDenseVectorObf-v0',
    'MineRLObtainIronPickaxeVectorObf-v0',
    'MineRLObtainIronPickaxeDenseVectorObf-v0',

    'MineRLBasaltFindCave-v0',
    'MineRLBasaltCreateVillageAnimalPen-v0',
    'MineRLBasaltMakeWaterfall-v0',
    'MineRLBasaltBuildVillageHouse-v0'
]

for env_id in ENVIRONMENTS:
    print(f"[DOWNLOAD] Preparing data for: {env_id}")
    try:
        data = minerl.data.make(env_id)
        print(f"[OK] Download complete or already cached for {env_id}")
    except Exception as e:
        print(f"[ERROR] Could not download {env_id}: {e}")

print("\n✅ All requested environments have been downloaded or cached.")

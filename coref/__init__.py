import os.path as osp
from dotenv import load_dotenv

load_dotenv(verbose=True)
COREF_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

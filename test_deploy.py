# test_destroy.py
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(message)s')

import config
config.AWS_REGION = 'eu-north-1'

from pathlib import Path
from agent.deployer import destroy

workspace = Path('terraform_workspace/manual-test-001')
result = destroy(workspace, 'thesis-test')
print(f"destroy: {'✓ ok' if result else '✗ ошибка'}")
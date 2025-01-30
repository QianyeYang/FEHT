import os, feht
from loguru import logger


def main():
    current_branch = os.popen(f'cd {feht.DIR_PROJECT} && git branch --show-current').read().strip()
    logger.info(f"To upgarde the feht package on branch {current_branch}...")
    
    # check if the local contains any changes
    changes = os.popen(f'cd {feht.DIR_PROJECT} && git status --porcelain').read().strip()
    if changes:
        logger.error("Local contains changes. Please commit/clean/stash them before upgrading.")
        print(changes)
        return
    
    # start upgrading
    os.system(f'cd {feht.DIR_PROJECT} && git pull')
    os.system(f'cd {feht.DIR_PROJECT} && pip install -e .')
    logger.info(f"Upgraded the feht package on branch {current_branch} successfully.")
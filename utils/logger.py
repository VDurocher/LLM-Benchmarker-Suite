"""
Logger structuré pour le suite LLM-Benchmarker.

Utilise le module standard `logging` avec un formatter personnalisé
qui produit des lignes lisibles en développement et JSON en production.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

LOG_FORMAT: Final[str] = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Niveau de log configurable via variable d'environnement
_LOG_LEVEL: str = os.environ.get("LLM_BENCH_LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger configuré avec le nom du module appelant.
    Idempotent : appeler plusieurs fois avec le même nom retourne le même logger.
    """
    logger = logging.getLogger(name)

    # Évite les handlers dupliqués si la fonction est appelée plusieurs fois
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    logger.propagate = False

    return logger

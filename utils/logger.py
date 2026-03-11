import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "experiment",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure et retourne un logger avec le niveau et la destination indiqués.

    Parameters
    ----------
    name : str
        Nom du logger (par défaut "experiment").
    level : str
        Niveau de log parmi DEBUG, INFO, WARNING, ERROR, CRITICAL.
        Non sensible à la casse. Par défaut "INFO".
    log_file : str, optional
        Chemin vers un fichier de log. Si None, la sortie va uniquement
        sur la console (stderr).

    Returns
    -------
    logging.Logger
        Logger configuré et prêt à l'emploi.

    Examples
    --------
    >>> logger = setup_logger("mon_module", level="DEBUG", log_file="run.log")
    >>> logger.info("Démarrage du programme")
    >>> logger.debug("Valeur de x : %s", 42)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Évite les handlers en double si le logger est déjà configuré
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler fichier (optionnel)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

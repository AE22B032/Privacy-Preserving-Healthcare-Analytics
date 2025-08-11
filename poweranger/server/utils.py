import logging
import os


def setup_logging(level: int | str = None) -> None:
	lvl = level or os.getenv("LOGLEVEL", "INFO").upper()
	logging.basicConfig(
		level=getattr(logging, str(lvl), logging.INFO),
		format="%(asctime)s %(levelname)s : %(message)s",
	)


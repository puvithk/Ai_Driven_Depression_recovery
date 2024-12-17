from celery import shared_task
from .utils import consolidate_moods

@shared_task
def run_consolidate_moods():
    consolidate_moods()
    return "Mood consolidation completed."

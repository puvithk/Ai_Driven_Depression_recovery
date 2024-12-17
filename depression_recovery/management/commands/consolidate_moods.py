from django.core.management.base import BaseCommand
from ...utils import consolidate_moods

class Command(BaseCommand):
    help = 'Consolidates moods from ChatLog to MoodText'

    def handle(self, *args, **kwargs):
        consolidate_moods()
        self.stdout.write("Mood consolidation completed.")

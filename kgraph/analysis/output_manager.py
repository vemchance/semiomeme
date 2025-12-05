from pathlib import Path
from datetime import datetime
import shutil


PROJECT_ROOT = Path(__file__).parent.parent


class AnalysisOutputManager:
    def __init__(self, base_dir: str = "outputs/analysis"):
        # Make base_dir relative to project root
        if not Path(base_dir).is_absolute():
            self.base_dir = PROJECT_ROOT / base_dir
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_analysis_session(self, prefix: str = "analysis") -> Path:
        """Creates timestamped folders in outputs/analysis/"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.base_dir / f"{prefix}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def cleanup_old_sessions(self, keep_recent: int = 3):
        """Removes old analysis folders"""
        if not self.base_dir.exists():
            return

        # Get all timestamped directories
        session_dirs = [d for d in self.base_dir.iterdir()
                        if d.is_dir() and '_' in d.name]

        # Sort by modification time (newest first)
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old sessions beyond keep_recent
        for old_session in session_dirs[keep_recent:]:
            try:
                shutil.rmtree(old_session)
                print(f"Removed old analysis session: {old_session.name}")
            except OSError as e:
                print(f"Could not remove {old_session.name}: {e}")
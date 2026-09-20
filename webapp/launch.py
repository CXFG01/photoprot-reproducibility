"""Start the one managed API service; never spawn a second GPU worker."""
import subprocess
subprocess.run(['sudo','systemctl','start','photoprot-api'],check=True)
subprocess.run(['systemctl','is-active','photoprot-api'],check=True)

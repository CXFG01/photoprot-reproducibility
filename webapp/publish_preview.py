"""Start the existing named tunnel. Quick tunnels are retired."""
import subprocess
subprocess.run(['sudo','systemctl','start','photoprot-tunnel'],check=True)
print('Website: https://photoprot.uk')
print('API: https://api.photoprot.uk')

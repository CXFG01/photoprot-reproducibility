# Persistent hosting on Brev

These units assume the existing full PhotoProt service checkout, its web assets,
Python environment and index are installed at `/home/shadeform/photoprot`.
The reproducibility repository alone is not a complete website deployment.

Create a named Cloudflare Tunnel. Configure only `photoprot.uk` and
`api.photoprot.uk` to reach `http://127.0.0.1:8000`. Store its secret token at
`/home/shadeform/.cloudflared/photoprot-token`, owned by shadeform with mode 0600;
the enclosing directory should be mode 0700. Never commit that token.

Install these unit files in `/etc/systemd/system`, run `sudo systemctl daemon-reload`,
then enable both with `sudo systemctl enable --now photoprot-api photoprot-tunnel`.
Before migration, gracefully stop the existing API process occupying port 8000.
Keep the previous tunnel available until HTTPS health and image-search tests pass.

The stable hostname survives tunnel restarts. Availability still depends on the
Brev instance remaining running and on its network/GPU capacity.

Current units include resource limits, read-only filesystem views and offline
model loading. The API requires `webapp/protection.py` and must keep
`--no-proxy-headers` so only the actual localhost tunnel peer can provide the
client identity. Read [the operations guide](../webapp/OPERATIONS.md) before
changing worker counts, rate limits or security settings. The two legacy Python
launchers now call these same system services instead of creating new processes.

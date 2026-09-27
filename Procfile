# Fallback start command for Render.
#
# Render uses this ONLY when the service's Start Command is left blank. If a
# start command is set in the dashboard - or in render.yaml - that wins and this
# file is ignored. It is here so that a service created with an empty start
# command still boots the app instead of falling back to a framework guess.
#
# The flags are not decorative:
#   --server.port $PORT        Render assigns the port at runtime; 8501 is
#                              almost certainly not it.
#   --server.address 0.0.0.0   The default is 127.0.0.1, which inside a
#                              container is reachable only from the container.
#                              The symptom is a service Render reports healthy
#                              while every external request times out.
#   --server.headless true     Without it Streamlit tries to open a browser,
#                              which is meaningless on a server.
web: streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false

# The Minecraft agents see the env server's full render (TASK line, RULES block, Craftable-now,
# history and the output-contract line all come from the server), which is also what the
# format-SFT agents were trained on. Any wrapper text here would shift the prompt away from that
# distribution, so the template is a pure passthrough.
MINECRAFT_BARE_TEMPLATE = """{task_description}"""

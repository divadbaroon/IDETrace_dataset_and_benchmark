"""Behavioral codes, event classification, and category mappings."""

BEHAVIORAL_CODES = {
    'thinking':     {'id': 'thinking',     'label': 'Thinking'},
    'implementing': {'id': 'implementing', 'label': 'Implementing'},
    'debugging':    {'id': 'debugging',    'label': 'Debugging'},
    'seekingHelp':  {'id': 'seekingHelp',  'label': 'Seeking Help'},
    'testing':      {'id': 'testing',      'label': 'Testing'},
    'unknown':      {'id': 'unknown',      'label': 'Unknown'},
}

CODE_EVENTS = {
    'CODE_TYPE', 'CODE_DELETE', 'CODE_DELETE_SELECTION',
    'CODE_PASTE', 'CODE_CUT', 'CODE_UNDO', 'CODE_REDO', 'CODE_INDENT',
    'CODE_UNKNOWN',
}

TERMINAL_EVENTS = {
    'TERMINAL_RUN', 'TEST_CASE_RESULT', 'TERMINAL_ERROR', 'TERMINAL_OUTPUT',
}

CHAT_INPUT_EVENTS = {
    'CHAT_TYPE', 'CHAT_PASTE', 'CHAT_DELETE', 'CHAT_QUERY',
}

CHAT_RESPONSE_EVENTS = {
    'CHAT_RESPONSE',
}

CATEGORY_TO_BEHAVIOR = {
    'code':         'implementing',
    'terminal':     'testing',
    'chatInput':    'seekingHelp',
    'chatResponse': 'thinking',
}


def get_behavior(behavior_id):
    """Look up a behavioral code by ID."""
    return BEHAVIORAL_CODES.get(behavior_id, BEHAVIORAL_CODES['unknown'])


def get_event_category(event):
    """Classify an event into a behavioral category."""
    event_type = event.get('type', '')
    if event_type in CODE_EVENTS:
        return 'code'
    if event_type in TERMINAL_EVENTS:
        return 'terminal'
    if event_type in CHAT_INPUT_EVENTS:
        return 'chatInput'
    if event_type in CHAT_RESPONSE_EVENTS:
        return 'chatResponse'
    return None


def get_behavior_for_category(category):
    """Map event category to behavioral state."""
    return get_behavior(CATEGORY_TO_BEHAVIOR.get(category, 'unknown'))
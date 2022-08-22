"""Representation utilities."""
import pathlib
import pprint

import numpy as np
import pretty_midi

import representation_utils
import utils

# Configuration
RESOLUTION = 12
MAX_BEAT = 1024
MAX_DURATION = 384  # Remember to modify known durations as well!

# Dimensions
# (NOTE: "type" must be the first dimension!)
# (NOTE: Remember to modify N_TOKENS as well!)
DIMENSIONS = [
    "type",
    "beat",
    "position",
    "instrument",
    "octave",
    "pitch",
    "duration",
]
assert DIMENSIONS[0] == "type"

# Type
TYPE_CODE_MAP = {
    "start-of-song": 0,
    "instrument": 1,
    "start-of-notes": 2,
    "note": 3,
    "end-of-song": 4,
}
CODE_TYPE_MAP = utils.inverse_dict(TYPE_CODE_MAP)

# Beat
BEAT_CODE_MAP = {i: i + 1 for i in range(MAX_BEAT + 1)}
BEAT_CODE_MAP[None] = 0
CODE_BEAT_MAP = utils.inverse_dict(BEAT_CODE_MAP)

# Position
POSITION_CODE_MAP = {i: i + 1 for i in range(RESOLUTION)}
POSITION_CODE_MAP[None] = 0
CODE_POSITION_MAP = utils.inverse_dict(POSITION_CODE_MAP)

# Octave
OCTAVE_CODE_MAP = {i: i + 1 for i in range(11)}
OCTAVE_CODE_MAP[None] = 0
CODE_OCTAVE_MAP = utils.inverse_dict(OCTAVE_CODE_MAP)

# Pitch
PITCH_CODE_MAP = {i: i + 1 for i in range(12)}
PITCH_CODE_MAP[None] = 0
CODE_PITCH_MAP = utils.inverse_dict(PITCH_CODE_MAP)

# Duration
KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
    120,
    144,
    168,
    192,
    384,
]
DURATION_CODE_MAP = {
    i: int(np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))) + 1
    for i in range(MAX_DURATION + 1)
}
DURATION_CODE_MAP[None] = 0
CODE_DURATION_MAP = {
    i + 1: duration for i, duration in enumerate(KNOWN_DURATIONS)
}

# Instrument
PROGRAM_INSTRUMENT_MAP = {
    # Pianos
    0: "piano",
    1: "piano",
    2: "piano",
    3: "piano",
    4: "electric-piano",
    5: "electric-piano",
    6: "harpsichord",
    7: "clavinet",
    # Chromatic Percussion
    8: "celesta",
    9: "glockenspiel",
    10: "music-box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    14: "tubular-bells",
    15: "dulcimer",
    # Organs
    16: "organ",
    17: "organ",
    18: "organ",
    19: "church-organ",
    20: "organ",
    21: "accordion",
    22: "harmonica",
    23: "bandoneon",
    # Guitars
    24: "nylon-string-guitar",
    25: "steel-string-guitar",
    26: "electric-guitar",
    27: "electric-guitar",
    28: "electric-guitar",
    29: "electric-guitar",
    30: "electric-guitar",
    31: "electric-guitar",
    # Basses
    32: "bass",
    33: "electric-bass",
    34: "electric-bass",
    35: "electric-bass",
    36: "slap-bass",
    37: "slap-bass",
    38: "synth-bass",
    39: "synth-bass",
    # Strings
    40: "violin",
    41: "viola",
    42: "cello",
    43: "contrabass",
    44: "strings",
    45: "strings",
    46: "harp",
    47: "timpani",
    # Ensemble
    48: "strings",
    49: "strings",
    50: "synth-strings",
    51: "synth-strings",
    52: "voices",
    53: "voices",
    54: "voices",
    55: "orchestra-hit",
    # Brass
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "trumpet",
    60: "horn",
    61: "brasses",
    62: "synth-brasses",
    63: "synth-brasses",
    # Reed
    64: "soprano-saxophone",
    65: "alto-saxophone",
    66: "tenor-saxophone",
    67: "baritone-saxophone",
    68: "oboe",
    69: "english-horn",
    70: "bassoon",
    71: "clarinet",
    # Pipe
    72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan-flute",
    76: None,
    77: None,
    78: None,
    79: "ocarina",
    # Synth Lead
    80: "lead",
    81: "lead",
    82: "lead",
    83: "lead",
    84: "lead",
    85: "lead",
    86: "lead",
    87: "lead",
    # Synth Pad
    88: "pad",
    89: "pad",
    90: "pad",
    91: "pad",
    92: "pad",
    93: "pad",
    94: "pad",
    95: "pad",
    # Synth Effects
    96: None,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    # Ethnic
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    107: "koto",
    108: "kalimba",
    109: "bag-pipe",
    110: "violin",
    111: "shehnai",
    # Percussive
    112: None,
    113: None,
    114: None,
    115: None,
    116: None,
    117: "melodic-tom",
    118: "synth-drums",
    119: "synth-drums",
    # Sound effects
    120: None,
    121: None,
    122: None,
    123: None,
    124: None,
    125: None,
    126: None,
    127: None,
}
INSTRUMENT_PROGRAM_MAP = {
    # Pianos
    "piano": 0,
    "electric-piano": 4,
    "harpsichord": 6,
    "clavinet": 7,
    # Chromatic Percussion
    "celesta": 8,
    "glockenspiel": 9,
    "music-box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular-bells": 14,
    "dulcimer": 15,
    # Organs
    "organ": 16,
    "church-organ": 19,
    "accordion": 21,
    "harmonica": 22,
    "bandoneon": 23,
    # Guitars
    "nylon-string-guitar": 24,
    "steel-string-guitar": 25,
    "electric-guitar": 26,
    # Basses
    "bass": 32,
    "electric-bass": 33,
    "slap-bass": 36,
    "synth-bass": 38,
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "harp": 46,
    "timpani": 47,
    # Ensemble
    "strings": 49,
    "synth-strings": 50,
    "voices": 52,
    "orchestra-hit": 55,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "horn": 60,
    "brasses": 61,
    "synth-brasses": 62,
    # Reed
    "soprano-saxophone": 64,
    "alto-saxophone": 65,
    "tenor-saxophone": 66,
    "baritone-saxophone": 67,
    "oboe": 68,
    "english-horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan-flute": 75,
    "ocarina": 79,
    # Synth Lead
    "lead": 80,
    # Synth Pad
    "pad": 88,
    # Ethnic
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bag-pipe": 109,
    "shehnai": 111,
    # Percussive
    "melodic-tom": 117,
    "synth-drums": 118,
}
KNOWN_PROGRAMS = list(
    k for k, v in PROGRAM_INSTRUMENT_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))
INSTRUMENT_CODE_MAP = {
    instrument: i + 1 for i, instrument in enumerate(KNOWN_INSTRUMENTS)
}
INSTRUMENT_CODE_MAP[None] = 0
CODE_INSTRUMENT_MAP = utils.inverse_dict(INSTRUMENT_CODE_MAP)

N_TOKENS = [
    max(TYPE_CODE_MAP.values()) + 1,
    max(BEAT_CODE_MAP.values()) + 1,
    max(POSITION_CODE_MAP.values()) + 1,
    max(INSTRUMENT_CODE_MAP.values()) + 1,
    max(OCTAVE_CODE_MAP.values()) + 1,
    max(PITCH_CODE_MAP.values()) + 1,
    max(DURATION_CODE_MAP.values()) + 1,
]


def get_encoding():
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_duration": MAX_DURATION,
        "dimensions": DIMENSIONS,
        "n_tokens": N_TOKENS,
        "type_code_map": TYPE_CODE_MAP,
        "beat_code_map": BEAT_CODE_MAP,
        "position_code_map": POSITION_CODE_MAP,
        "instrument_code_map": INSTRUMENT_CODE_MAP,
        "octave_code_map": OCTAVE_CODE_MAP,
        "pitch_code_map": PITCH_CODE_MAP,
        "duration_code_map": DURATION_CODE_MAP,
        "code_type_map": CODE_TYPE_MAP,
        "code_beat_map": CODE_BEAT_MAP,
        "code_position_map": CODE_POSITION_MAP,
        "code_instrument_map": CODE_INSTRUMENT_MAP,
        "code_octave_map": CODE_OCTAVE_MAP,
        "code_pitch_map": CODE_PITCH_MAP,
        "code_duration_map": CODE_DURATION_MAP,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
    }


def load_encoding(filename):
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename)
    for key in (
        "code_type_map",
        "code_beat_map",
        "code_position_map",
        "code_instrument_map",
        "code_octave_map",
        "code_pitch_map",
        "code_duration_map",
        "beat_code_map",
        "position_code_map",
        "octave_code_map",
        "pitch_code_map",
        "duration_code_map",
        "program_instrument_map",
    ):
        encoding[key] = {
            int(k) if k != "null" else None: v
            for k, v in encoding[key].items()
        }
    return encoding


def encode_notes(notes, encoding):
    """Encode a note sequence into a sequence of codes.

    Each row of the input is a note specified as follows.

        (beat, position, pitch, duration, program)

    Each row of the output is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get variables
    max_beat = encoding["max_beat"]
    max_duration = encoding["max_duration"]

    # Get maps
    type_code_map = encoding["type_code_map"]
    beat_code_map = encoding["beat_code_map"]
    position_code_map = encoding["position_code_map"]
    instrument_code_map = encoding["instrument_code_map"]
    octave_code_map = encoding["octave_code_map"]
    pitch_code_map = encoding["pitch_code_map"]
    duration_code_map = encoding["duration_code_map"]
    program_instrument_map = encoding["program_instrument_map"]

    # Get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    instrument_dim = encoding["dimensions"].index("instrument")
    octave_dim = encoding["dimensions"].index("octave")
    pitch_dim = encoding["dimensions"].index("pitch")
    duration_dim = encoding["dimensions"].index("duration")

    # Start the codes with an SOS row
    codes = [[type_code_map["start-of-song"], 0, 0, 0, 0, 0, 0]]

    # Extract instruments
    instruments = set(program_instrument_map[note[-1]] for note in notes)

    # Encode the instruments
    instrument_codes = []
    for instrument in instruments:
        # Skip unknown instruments
        if instrument is None:
            continue
        row = [type_code_map["instrument"], 0, 0, 0, 0, 0, 0]
        row[instrument_dim] = instrument_code_map[instrument]
        instrument_codes.append(row)

    # Sort the instruments and append them to the code sequence
    instrument_codes.sort()
    codes.extend(instrument_codes)

    # Encode the notes
    codes.append([type_code_map["start-of-notes"], 0, 0, 0, 0, 0, 0])
    for beat, position, pitch_, duration, program in notes:
        # Skip if max_beat has reached
        if beat > max_beat:
            continue
        # Skip unknown instruments
        instrument = program_instrument_map[program]
        if instrument is None:
            continue
        # Encode the note
        row = [type_code_map["note"], 0, 0, 0, 0, 0, 0]
        row[beat_dim] = beat_code_map[beat]
        row[position_dim] = position_code_map[position]
        row[instrument_dim] = instrument_code_map[instrument]
        octave, pitch = divmod(pitch_, 12)
        row[octave_dim] = octave_code_map[octave]
        row[pitch_dim] = pitch_code_map[pitch]
        row[duration_dim] = duration_code_map[min(duration, max_duration)]
        codes.append(row)

    # End the codes with an EOS row
    codes.append([type_code_map["end-of-song"], 0, 0, 0, 0, 0, 0])

    # Sort the codes
    codes.sort()

    return np.array(codes)


def encode(music, encoding):
    """Encode a MusPy music object into a sequence of codes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    Each row of the output is a note specified as follows.

        (beat, position, pitch, duration, program)

    """
    # Extract notes
    notes = representation_utils.extract_notes(music, encoding["resolution"])

    # Encode the notes
    codes = encode_notes(notes, encoding)

    return codes


def decode_notes(codes, encoding):
    """Decode codes into a note sequence.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get variables and maps
    code_type_map = encoding["code_type_map"]
    code_beat_map = encoding["code_beat_map"]
    code_position_map = encoding["code_position_map"]
    code_instrument_map = encoding["code_instrument_map"]
    code_octave_map = encoding["code_octave_map"]
    code_pitch_map = encoding["code_pitch_map"]
    code_duration_map = encoding["code_duration_map"]
    instrument_program_map = encoding["instrument_program_map"]

    # Get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    instrument_dim = encoding["dimensions"].index("instrument")
    octave_dim = encoding["dimensions"].index("octave")
    pitch_dim = encoding["dimensions"].index("pitch")
    duration_dim = encoding["dimensions"].index("duration")

    # Decode the codes into a sequence of notes
    notes = []
    for row in codes:
        event_type = code_type_map[int(row[0])]
        if event_type in ("start-of-song", "instrument", "start-of-notes"):
            continue
        elif event_type == "end-of-song":
            break
        elif event_type == "note":
            beat = code_beat_map[int(row[beat_dim])]
            position = code_position_map[int(row[position_dim])]
            instrument = code_instrument_map[int(row[instrument_dim])]
            program = instrument_program_map[instrument]
            octave = code_octave_map[int(row[octave_dim])]
            pitch = code_pitch_map[int(row[pitch_dim])]
            duration = code_duration_map[int(row[duration_dim])]
            notes.append(
                (beat, position, 12 * octave + pitch, duration, program)
            )
        else:
            raise ValueError("Unknown event type.")

    return notes


def decode(codes, encoding):
    """Decode codes into a MusPy Music object.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get resolution
    resolution = encoding["resolution"]

    # Decode codes into a note sequence
    notes = decode_notes(codes, encoding)

    # Reconstruct the music object
    music = representation_utils.reconstruct(notes, resolution)

    return music


def dump(data, encoding):
    """Decode the codes and dump as a string."""
    # Get maps
    code_type_map = encoding["code_type_map"]
    code_beat_map = encoding["code_beat_map"]
    code_position_map = encoding["code_position_map"]
    code_instrument_map = encoding["code_instrument_map"]
    code_octave_map = encoding["code_octave_map"]
    code_pitch_map = encoding["code_pitch_map"]
    code_duration_map = encoding["code_duration_map"]

    # Get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    instrument_dim = encoding["dimensions"].index("instrument")
    octave_dim = encoding["dimensions"].index("octave")
    pitch_dim = encoding["dimensions"].index("pitch")
    duration_dim = encoding["dimensions"].index("duration")

    # Iterate over the rows
    lines = []
    for row in data:
        event_type = code_type_map[int(row[0])]
        if event_type == "start-of-song":
            lines.append("Start of song")
        elif event_type == "end-of-song":
            lines.append("End of song")
        elif event_type == "instrument":
            instrument = code_instrument_map[int(row[instrument_dim])]
            lines.append(f"Instrument: {instrument}")
        elif event_type == "start-of-notes":
            lines.append("Start of notes")
        elif event_type == "note":
            beat = code_beat_map[int(row[beat_dim])]
            position = code_position_map[int(row[position_dim])]
            instrument = code_instrument_map[int(row[instrument_dim])]
            pitch = pretty_midi.note_number_to_name(
                12 * code_octave_map[int(row[octave_dim])]
                + code_pitch_map[int(row[pitch_dim])]
            )
            duration = code_duration_map[int(row[duration_dim])]
            lines.append(
                f"Note: beat={beat + 1}, position={position + 1}, "
                f"instrument={instrument}, pitch={pitch}, duration={duration}"
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    return "\n".join(lines)


def save_txt(filename, data, encoding):
    """Dump the codes into a TXT file."""
    with open(filename, "w") as f:
        f.write(dump(data, encoding))


def save_csv_codes(filename, data):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 7
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="type,beat,position,instrument,octave,pitch,duration",
        comments="",
    )


def main():
    """Main function."""
    # Get the encoding
    encoding = get_encoding()

    # Save the encoding
    filename = pathlib.Path(__file__).parent / "encoding.json"
    utils.save_json(filename, encoding)

    # Load encoding
    encoding = load_encoding(filename)

    # Print the maps
    print(f"{' Maps ':=^40}")
    for key, value in encoding.items():
        if key in (
            "instrument_code_map",
            "code_instrument_map",
            "program_instrument_map",
            "instrument_program_map",
        ):
            print("-" * 40)
            print(f"{key}:")
            pprint.pprint(value, indent=2)

    # Print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_beat: {encoding['max_beat']}")
    print(f"max_duration: {encoding['max_duration']}")

    # Print the number of tokens
    print(f"{' Number of tokens ':=^40}")
    keys = (
        "type",
        "beat",
        "position",
        "instrument",
        "octave",
        "pitch",
        "duration",
    )
    for key, value in zip(keys, N_TOKENS):
        print(f"{key}: {value}")

    # Print an example
    print(f"{'Example':=^40}")
    codes = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 3, 0, 0, 0),
            (1, 0, 0, 33, 0, 0, 0),
            (2, 0, 0, 0, 0, 0, 0),
            (3, 1, 1, 3, 4, 1, 15),
            (3, 1, 1, 3, 5, 1, 15),
            (3, 1, 1, 3, 5, 5, 15),
            (3, 1, 1, 3, 5, 8, 15),
            (3, 1, 1, 33, 5, 8, 10),
            (3, 2, 1, 33, 5, 8, 10),
            (3, 3, 1, 33, 5, 8, 10),
            (3, 4, 1, 3, 5, 1, 15),
            (3, 4, 1, 3, 5, 5, 10),
            (3, 4, 1, 3, 5, 8, 10),
            (3, 4, 1, 3, 6, 1, 10),
            (3, 4, 1, 33, 5, 1, 10),
            (3, 4, 1, 33, 5, 5, 4),
            (3, 4, 7, 33, 5, 3, 4),
            (4, 0, 0, 0, 0, 0, 0),
        ),
        int,
    )
    print(f"Codes:\n{codes}")

    print("-" * 40)
    print(f"Decoded:\n{dump(codes, encoding)}")

    music = decode(codes, encoding)
    print("-" * 40)
    print(f"Decoded music:\n{music}")

    encoded = encode(music, encoding)
    print("-" * 40)
    print(f"Encoded:\n{encoded}")
    assert np.all(codes == encoded)


if __name__ == "__main__":
    main()

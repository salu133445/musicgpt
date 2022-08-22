"""Representation utilities."""
import muspy
import numpy as np


def extract_notes(music, resolution):
    """Return a MusPy music object as a note sequence.

    Each row of the output is a note specified as follows.

        (beat, position, pitch, duration, program)

    """
    # Check resolution
    assert music.resolution == resolution

    # Extract notes
    notes = []
    for track in music:
        # Skip track with a nonstandard program number
        if track.program < 0 or track.program > 127:
            continue
        program = track.program if not track.is_drum else 128
        for note in track:
            if note.duration < 0:
                continue
            beat, position = divmod(note.time, resolution)
            duration = max(note.duration, 1)
            notes.append((beat, position, note.pitch, duration, program))

    # Deduplicate and sort the notes
    notes = sorted(set(notes))

    return np.array(notes)


def reconstruct(notes, resolution):
    """Reconstruct a note sequence to a MusPy Music object."""
    # Construct the MusPy Music object
    music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(0, 100)])

    # Append the tracks
    programs = sorted(set(note[-1] for note in notes))
    for program in programs:
        music.tracks.append(muspy.Track(program))

    # Append the notes
    for beat, position, pitch, duration, program in notes:
        time = beat * resolution + position
        track_idx = programs.index(program)
        music[track_idx].notes.append(muspy.Note(time, pitch, duration))

    return music


def save_csv_notes(filename, data):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 5
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="beat,position,pitch,duration,program",
        comments="",
    )

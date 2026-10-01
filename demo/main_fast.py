"""Same TSE flip-angle optimization as ``main.py``, with the conversion cache on.

Each iteration rebuilds the sequence with a new refocusing-flip tensor and the
same blocks. ``speed_up_by_assuming_const_seq_structure`` refreshes those
tensor values instead of converting the layout again.
"""

from main import main


if __name__ == '__main__':
    main(const_structure=True)

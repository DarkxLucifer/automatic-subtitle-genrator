from datetime import timedelta

def seconds_to_srt_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def segments_to_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = seconds_to_srt_timestamp(seg["start"])
        end_ts = seconds_to_srt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)

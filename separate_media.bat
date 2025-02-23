@echo off
set input_file=input/input.mp4
set output_video=output/%~n1_video_only.mov
set output_audio=output/%~n1_audio_only.mp4

echo Separating video-only stream...
ffmpeg -i %input_file% -an -c:v copy %output_video%

echo Separating audio-only stream...
ffmpeg -i %input_file% -vn -c:a copy %output_audio%

echo Separation complete!
pause

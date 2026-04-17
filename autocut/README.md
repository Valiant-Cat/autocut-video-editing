# autocut

一个自动视频剪辑流水线，只有一个主入口。

- 输入：`video_path`、`audio_path`、`prompt`
- 输出：一个最终成片视频

核心流程：

1. 将源视频切分为候选片段
2. 使用视觉模型描述每个候选片段
3. 使用音频模型总结配乐结构
4. 使用 Agent 模型为每个音频槽位选择片段
5. 使用 `ffmpeg` 渲染最终成片

## 文件说明

- `run.py`：函数入口与 CLI 入口
- `config.py`：`.env` 加载与运行配置
- `llm_clients.py`：Gemini 与 OpenAI 兼容客户端
- `media.py`：`ffmpeg` / `ffprobe` 工具封装
- `pipeline.py`：端到端流程编排

## 运行要求

- Python 3.10+
- `ffmpeg`
- `ffprobe`

## 使用方法

```bash
cd D:/111-123/autocut-video-editing/assets/autocut
python run.py \
  --video-path D:/data/source.mp4 \
  --audio-path D:/data/music.mp3 \
  --prompt "剪一个情绪递进、节奏明确、适合音乐卡点的版本"
```

Python 调用示例：

```python
from run import run

result = run(
    video_path="/abs/path/video.mp4",
    audio_path="/abs/path/audio.mp3",
    prompt="剪一个节奏递进、情绪拉满的版本",
)

print(result["final_video"])
```

输出目录位于 `output/<job_name>/`。

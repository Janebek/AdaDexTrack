# GitHub Pages 本地视频相册（模板）

## 使用
1. 把本项目的所有文件放到你的仓库根目录（推荐仓库名：`你的用户名.github.io` 或任何项目仓库）。
2. 将你的 `.mp4` 视频放进 `videos/` 目录（可选：把同名 `.jpg/.png` 放到 `thumbs/` 作为海报图）。
3. 提交并推送到 `main` 分支。
4. 在仓库 **Settings → Pages** 中，将 **Build and deployment → Source** 设置为 **GitHub Actions**。
5. Actions 运行结束后，访问 Pages 提供的 URL 即可看到相册。

> 每次你往 `videos/` 新增或删除文件，工作流都会自动重新生成 `manifest.json` 并部署。

test a

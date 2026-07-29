import { promises as fs } from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const VIDEO_DIR = path.join(ROOT, 'videos');
const THUMB_DIR = path.join(ROOT, 'thumbs');
const OUT = path.join(ROOT, 'manifest.json');

const VIDEO_EXTS = new Set(['.mp4', '.webm', '.m4v']); // 建议使用 .mp4(H.264/AAC)

function toTitle(name){
  return name.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

async function exists(p){
  try { await fs.access(p); return true; } catch { return false; }
}

async function main(){
  const entries = await fs.readdir(VIDEO_DIR, { withFileTypes: true });
  const files = entries.filter(e => e.isFile()).map(e => e.name).filter(n => VIDEO_EXTS.has(path.extname(n).toLowerCase()));
  files.sort((a,b) => a.localeCompare(b, 'zh-Hans-CN'));

  const list = [];
  for(const filename of files){
    const base = filename.replace(/\.[^.]+$/, '');
    const posterJpg = path.join(THUMB_DIR, base + '.jpg');
    const posterPng = path.join(THUMB_DIR, base + '.png');
    const poster = (await exists(posterJpg)) ? `thumbs/${base}.jpg` : (await exists(posterPng)) ? `thumbs/${base}.png` : '';
    list.push({
      file: filename,
      src: `videos/${filename}`,
      title: toTitle(base),
      poster,
      desc: ''
    });
  }

  await fs.writeFile(OUT, JSON.stringify(list, null, 2), 'utf8');
  console.log(`✔ 写入 ${OUT}，共 ${list.length} 条`);
}

main().catch(err => { console.error(err); process.exit(1); });

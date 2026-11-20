import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const jsonPath = path.resolve(process.cwd(), "..", "benchmark_results_58d.json");
    if (fs.existsSync(jsonPath)) {
      const raw = fs.readFileSync(jsonPath, "utf-8");
      return NextResponse.json(JSON.parse(raw));
    }
    return NextResponse.json({ error: "benchmark_results_58d.json not found" }, { status: 404 });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}

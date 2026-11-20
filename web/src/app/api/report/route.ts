import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const pdfPath = path.resolve(process.cwd(), "..", "kmeans_convergence_report.pdf");
    if (fs.existsSync(pdfPath)) {
      const buf = fs.readFileSync(pdfPath);
      return new NextResponse(buf, {
        status: 200,
        headers: {
          "Content-Type": "application/pdf",
          "Content-Disposition": 'inline; filename="kmeans_convergence_report.pdf"',
        },
      });
    }
    return NextResponse.json({ error: "PDF report not found" }, { status: 404 });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}

import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  weight: ["400", "500", "600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Mini-Projet 10 — Clustering K-Means : RDD (L1/L2) vs PySpark MLlib",
  description:
    "Benchmark comparatif distribué : K-Means Manuel sur RDDs (Distances L1/L2, reduceByKey, WSSE) vs PySpark MLlib (K-Means||, Elbow, Silhouette, PCA 2D) sur vecteurs de documents 58D.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="fr" className={`${dmSans.variable} ${jetbrainsMono.variable}`}>
      <body className="bg-zinc-50 dark:bg-[#09090b] text-zinc-950 dark:text-zinc-50 font-sans antialiased min-h-screen transition-colors duration-200">
        {children}
      </body>
    </html>
  );
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/benchmark/stream",
        destination: "http://127.0.0.1:8050/api/benchmark/stream",
      },
      {
        source: "/kmeans_convergence_report.pdf",
        destination: "http://127.0.0.1:8050/kmeans_convergence_report.pdf",
      },
    ];
  },
};

export default nextConfig;

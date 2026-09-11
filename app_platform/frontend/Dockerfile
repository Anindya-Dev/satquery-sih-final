FROM node:18-alpine

WORKDIR /app

COPY app_platform/frontend/package*.json ./
RUN npm install

COPY app_platform/frontend/ ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]